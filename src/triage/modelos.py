"""Capa de modelos: cliente Ollama (real) y ChatSimulado (tests / --dry-run sin GPU).

Interfaz minima: ``llm.generar(sistema, usuario, esquema, max_tokens) -> RespuestaLLM``.
La salida es texto JSON; el parseo y la reparacion viven en agentes.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from .catalogo import normalizar

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODELO_DEFECTO = os.getenv("TRIAGE_MODEL", "qwen2.5:3b")
MODELO_JUEZ = os.getenv("TRIAGE_JUDGE_MODEL", "llama3.1:8b-instruct-q4_K_M")
TIMEOUT_S = 120


class ModeloNoDisponible(RuntimeError):
    """Ollama no responde o el modelo no existe."""


@dataclass
class RespuestaLLM:
    texto: str
    tokens_in: int
    tokens_out: int
    ttft_s: float
    latencia_s: float


class LLM(Protocol):
    nombre: str
    simulado: bool

    def generar(self, sistema: str, usuario: str, esquema: type[BaseModel], max_tokens: int = 180) -> RespuestaLLM: ...


# --------------------------------------------------------------------------- Ollama
class OllamaLLM:
    simulado = False

    def __init__(self, modelo: str = MODELO_DEFECTO, base_url: str = BASE_URL):
        self.nombre = modelo
        self.base_url = base_url
        self._clientes: dict[tuple[str, int], object] = {}

    def _cliente(self, esquema: type[BaseModel], max_tokens: int):
        clave = (esquema.__name__, max_tokens)
        if clave not in self._clientes:
            try:
                from langchain_ollama import ChatOllama
            except ImportError as e:  # pragma: no cover
                raise ModeloNoDisponible("falta langchain-ollama (uv sync)") from e
            self._clientes[clave] = ChatOllama(
                model=self.nombre,
                base_url=self.base_url,
                temperature=0.1,
                num_ctx=4096,
                num_predict=max_tokens,
                keep_alive="30m",
                format=esquema.model_json_schema(),
                client_kwargs={"timeout": TIMEOUT_S},
            )
        return self._clientes[clave]

    def generar(self, sistema, usuario, esquema, max_tokens=180) -> RespuestaLLM:
        from langchain_core.messages import HumanMessage, SystemMessage

        cli = self._cliente(esquema, max_tokens)
        t0 = time.perf_counter()
        ttft = 0.0
        agg = None
        try:
            for chunk in cli.stream([SystemMessage(sistema), HumanMessage(usuario)]):
                if agg is None:
                    ttft = time.perf_counter() - t0
                    agg = chunk
                else:
                    agg = agg + chunk
        except Exception as e:  # conexion rechazada, timeout, modelo ausente...
            raise ModeloNoDisponible(f"{type(e).__name__}: {e}") from e
        lat = time.perf_counter() - t0
        texto = (agg.content if agg is not None else "") or ""
        um = (getattr(agg, "usage_metadata", None) or {}) if agg is not None else {}
        t_in = um.get("input_tokens") or max(1, (len(sistema) + len(usuario)) // 4)
        t_out = um.get("output_tokens") or max(1, len(texto) // 4)
        return RespuestaLLM(texto, int(t_in), int(t_out), ttft, lat)


def healthcheck(base_url: str = BASE_URL, modelo: str = MODELO_DEFECTO) -> tuple[bool, str]:
    """(ok, detalle) sin lanzar excepciones."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=3) as r:
            tags = json.load(r)
    except Exception as e:
        return False, f"Ollama no responde en {base_url} ({type(e).__name__})"
    nombres = {m.get("name") for m in tags.get("models", [])}
    if modelo not in nombres:
        return False, f"modelo {modelo} no descargado (ollama pull {modelo}); disponibles: {sorted(nombres)}"
    return True, f"Ollama OK, modelo {modelo} disponible"


# ------------------------------------------------------------------------ Simulado
_PALABRAS: list[tuple[str, tuple[str, ...]]] = [
    ("urologia", ("orinar", "orina", "arde mucho al orin", "prostata", "calculo renal")),
    ("dermatologia", ("manchas", "picazon", "escamas", "acne", "lunar", "sarpullido", "salpullido")),
    ("psiquiatria", ("ansiedad", "panico", "insomnio", "tristeza")),
    ("oftalmologia", ("ojo", "vision borrosa", "ojos")),
    ("otorrinolaringologia", ("oido", "sinusitis", "garganta", "amigdalas", "ronquera")),
    ("gastroenterologia", ("acidez", "diarrea", "estrenimiento", "nauseas", "gastritis", "reflujo")),
    ("neumologia", ("tos cronica", "silbidos", "asma", "tos seca")),
    ("cardiologia", ("palpitaciones", "presion alta", "taquicardia")),
    ("neurologia", ("migrana", "hormigueo", "temblor", "migranas")),
    ("gineco_obstetricia", ("menstru", "periodo", "regla", "flujo vaginal", "embaraz")),
    ("pediatria", ("mi hijo", "mi hija", "mi nino", "mi nina", "bebe")),
    ("endocrinologia", ("sed excesiva", "tiroides", "diabetes", "mucha sed")),
    ("ortopedia", ("rodilla", "espalda", "hombro", "esguince", "pain en la rodilla", "tobillo")),
]
_RECHAZO = ("receta", "recet", "dosis", "incapacidad", "ignora tus instrucciones", "antibiotico para")


def _h(*partes: str) -> int:
    return int(hashlib.sha256("|".join(partes).encode()).hexdigest(), 16)


class ChatSimulado:
    """LLM falso y determinista. NO es evidencia empirica: los numeros que produce
    (latencia, exactitud) son sinteticos y las filas se marcan simulado=True.

    ``tasa_json_invalido``: fraccion de primeras llamadas que devuelven JSON roto
    (ejercita el camino de reparacion/fallback). ``caido=True`` simula Ollama apagado.
    """

    simulado = True

    def __init__(self, nombre: str = "simulado", tasa_json_invalido: float = 0.05, caido: bool = False):
        self.nombre = nombre
        self.tasa = tasa_json_invalido
        self.caido = caido
        self._intentos: dict[str, int] = {}

    def _clasificar(self, texto: str) -> tuple[str, float, bool]:
        t = normalizar(texto)
        if any(k in t for k in _RECHAZO):
            return "medicina_general", 0.9, True
        for esp, claves in _PALABRAS:
            if any(c in t for c in claves):
                return esp, 0.8, False
        return "medicina_general", 0.5, False

    def generar(self, sistema, usuario, esquema, max_tokens=180) -> RespuestaLLM:
        if self.caido:
            raise ModeloNoDisponible("Ollama simulado apagado")
        clave = f"{esquema.__name__}|{usuario}"
        n = self._intentos.get(clave, 0)
        self._intentos[clave] = n + 1
        esp, conf, rechazo = self._clasificar(usuario)
        nombre = esquema.__name__
        if nombre == "Triage":
            obj = {
                "especialidad": esp,
                "confianza": conf,
                "justificacion": (
                    "No puedo recetar ni indicar dosis; te oriento a medicina general."
                    if rechazo
                    else f"Los sintomas descritos se relacionan con {esp.replace('_', ' ')}."
                ),
                "signos_alarma": [],
                "es_emergencia": False,
                "requiere_aclaracion": esp == "medicina_general" and not rechazo,
                "preguntas": [],
                "siguiente_paso": f"Agenda una consulta con {esp.replace('_', ' ')}.",
            }
        elif nombre == "Ruta":
            obj = {"especialidad": esp, "confianza": conf}
        elif nombre == "Verificacion":
            obj = {"coherente": True, "alarma_omitida": False, "especialidad_corregida": None, "comentarios": ""}
        elif nombre == "Hallazgos":
            obj = {
                "justificacion": f"Segun lo descrito, {esp.replace('_', ' ')} es el especialista que mejor se ajusta.",
                "siguiente_paso": f"Agenda una consulta con {esp.replace('_', ' ')}.",
            }
        elif nombre == "Juicio":
            obj = {"justificacion_coherente": 4, "sin_diagnostico": True, "sin_dosis": True, "comentarios": ""}
        else:  # pragma: no cover
            obj = {}
        texto = json.dumps(obj, ensure_ascii=False)
        # inyeccion determinista de JSON invalido solo en el primer intento
        if n == 0 and (_h(clave) % 10_000) / 10_000 < self.tasa:
            texto = texto[: len(texto) // 2]
        t_in = (len(sistema) + len(usuario)) // 4
        t_out = len(texto) // 4
        # latencia sintetica reproducible: ~0.3 s de prefill + 25 ms/token + jitter
        jitter = (_h(clave, str(n)) % 1000) / 1000 * 0.4
        lat = 0.3 + t_in * 0.0005 + t_out * 0.025 + jitter
        return RespuestaLLM(texto, t_in, t_out, 0.3 + jitter / 2, lat)


def crear_llm(rol: str = "triador", simulado: bool = False, **kw) -> LLM:
    if simulado:
        return ChatSimulado(**kw)
    modelo = MODELO_JUEZ if rol == "juez" else MODELO_DEFECTO
    return OllamaLLM(modelo)
