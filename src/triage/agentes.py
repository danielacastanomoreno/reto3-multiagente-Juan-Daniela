"""Agentes con LLM: A1 triador, A3 verificador, A4 router, A5 especialista.

Todos pasan por ``llamar``: JSON -> Pydantic, 1 reintento de reparacion, y se
registra una fila de traza por llamada. Si tras el reintento sigue invalido,
devuelve None y el sistema aplica fallback a MEDICINA_GENERAL.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .catalogo import catalogo_comprimido
from .esquemas import Hallazgos, Ruta, Triage, Verificacion
from .modelos import LLM, ModeloNoDisponible
from .prompts import (
    INSTR_ESPECIALISTA,
    INSTR_REPARACION,
    INSTR_ROUTER,
    INSTR_TRIADOR,
    INSTR_VERIFICADOR,
)

T = TypeVar("T", bound=BaseModel)

COSTO_HORA_COP = float(os.getenv("COSTO_HORA_EQUIPO_COP", "175"))  # 0.25 kW * 700 COP/kWh
TRM = float(os.getenv("TRM_COP_USD", "4100"))


def costo_usd(segundos: float) -> float:
    return segundos / 3600 * COSTO_HORA_COP / TRM


class Trazador:
    """Acumula una fila por llamada (esquema de plan §6.3)."""

    def __init__(self) -> None:
        self.filas: list[dict] = []

    def registrar(self, ctx: "Contexto", agente: str, modelo: str, tokens_in: int, tokens_out: int,
                  ttft_s: float, latencia_s: float, exito: bool, error: str = "",
                  llamadas: int = 1, simulado: bool = False) -> dict:
        fila = {
            "tarea_id": ctx.tarea_id, "config": ctx.config, "rep": ctx.rep, "agente": agente,
            "modelo": modelo, "tokens_in": tokens_in, "tokens_out": tokens_out,
            "ttft_s": round(ttft_s, 4), "latencia_s": round(latencia_s, 4), "exito": exito,
            "error": error, "especialidad_pred": "", "especialidad_esperada": ctx.esperado or "",
            "emergencia_pred": None, "emergencia_real": ctx.emergencia_real, "llamadas": llamadas,
            "costo_estimado_usd": round(costo_usd(latencia_s), 8), "fallback": False,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "simulado": simulado,
        }
        self.filas.append(fila)
        return fila


class Contexto:
    def __init__(self, tarea_id: str, config: str, rep: int, esperado: str | None,
                 emergencia_real: bool | None, trazador: Trazador):
        self.tarea_id, self.config, self.rep = tarea_id, config, rep
        self.esperado, self.emergencia_real, self.trazador = esperado, emergencia_real, trazador


def _extraer_json(texto: str) -> dict:
    ini, fin = texto.find("{"), texto.rfind("}")
    if ini < 0 or fin <= ini:
        raise ValueError("sin JSON")
    return json.loads(texto[ini : fin + 1])


def llamar(llm: LLM, ctx: Contexto, agente: str, sistema: str, usuario: str,
           esquema: type[T], max_tokens: int = 180) -> T | None:
    """Llama al LLM con 1 reintento de reparacion. Propaga ModeloNoDisponible."""
    for intento in range(2):
        sis = sistema if intento == 0 else sistema + "\n\n" + INSTR_REPARACION
        resp = llm.generar(sis, usuario, esquema, max_tokens)
        try:
            obj = esquema.model_validate(_extraer_json(resp.texto))
        except (ValueError, ValidationError) as e:
            ctx.trazador.registrar(ctx, agente, llm.nombre, resp.tokens_in, resp.tokens_out,
                                   resp.ttft_s, resp.latencia_s, False,
                                   f"{type(e).__name__}", simulado=llm.simulado)
            continue
        ctx.trazador.registrar(ctx, agente, llm.nombre, resp.tokens_in, resp.tokens_out,
                               resp.ttft_s, resp.latencia_s, True, simulado=llm.simulado)
        return obj
    return None


def triador(llm: LLM, ctx: Contexto, texto: str) -> Triage | None:  # A1
    return llamar(llm, ctx, "A1", INSTR_TRIADOR.format(catalogo=catalogo_comprimido()), texto, Triage, 180)


def router(llm: LLM, ctx: Contexto, texto: str) -> Ruta | None:  # A4
    return llamar(llm, ctx, "A4", INSTR_ROUTER.format(catalogo=catalogo_comprimido()), texto, Ruta, 60)


def especialista(llm: LLM, ctx: Contexto, texto: str, especialidad: str) -> Hallazgos | None:  # A5
    sis = INSTR_ESPECIALISTA.format(especialidad=especialidad.replace("_", " "))
    return llamar(llm, ctx, "A5", sis, texto, Hallazgos, 140)


def verificador(llm: LLM, ctx: Contexto, texto: str, t: Triage) -> Verificacion | None:  # A3
    usuario = f"SINTOMAS:\n{texto}\n\nSALIDA DEL TRIADOR:\n{t.model_dump_json()}"
    return llamar(llm, ctx, "A3", INSTR_VERIFICADOR, usuario, Verificacion, 100)
