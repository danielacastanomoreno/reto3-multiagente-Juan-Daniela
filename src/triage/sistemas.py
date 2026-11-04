"""Configuraciones A/B/C y ejecucion de una tarea de triaje.

A: A0 -> A1 -> A2                     (1 llamada LLM)
B: A0 -> A4 -> A3 -> A2               (2 llamadas)
C: A0 -> A4 -> A5 -> A3 -> A2         (3 llamadas)

La salida es siempre un ``TareaResultado`` con el mismo esquema, para comparar
fila a fila. Si la guardia dispara no se llama al modelo. Si Ollama no responde
se degrada a urgencias (si hubo alarma) o medicina general con mensaje claro.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .agentes import Contexto, Trazador, costo_usd, especialista, router, triador, verificador
from .catalogo import ESPECIALIDADES
from .esquemas import Especialidad, Triage, Verificacion
from .guardia import evaluar, fuera_de_alcance
from .modelos import LLM, ModeloNoDisponible
from .presentador import presentar, triage_urgencias

CONFIGS = ("A", "B", "C")
MAX_CHARS = 4000  # ~1000 tokens; deja margen en num_ctx=4096
AVISO_RECHAZO = (
    " No puedo recetar, indicar dosis ni dar incapacidades; consulta con un profesional de medicina general."
)


@dataclass
class TareaResultado:
    tarea_id: str
    config: str
    rep: int
    triage: Triage
    texto_respuesta: str
    exito: bool = True
    error: str = ""
    fallback: bool = False
    degradado: bool = False
    guardia_dispara: bool = False
    latencia_total_s: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    llamadas: int = 0
    latencia_coord_s: float = 0.0
    tokens_coord: int = 0
    simulado: bool = False
    filas: list[dict] = field(default_factory=list)

    @property
    def emergencia_pred(self) -> bool:
        return self.triage.es_emergencia or self.triage.especialidad == Especialidad.URGENCIAS


def _triage_basico(esp: Especialidad, conf: float, justificacion: str | None, paso: str | None) -> Triage:
    nombre = esp.value.replace("_", " ")
    info = ESPECIALIDADES[esp.value]
    return Triage(
        especialidad=esp,
        confianza=conf,
        justificacion=(justificacion or f"Segun lo que describes, {nombre} es el especialista que mejor se ajusta "
                       f"({info['cuando_derivar']}).")[:400],
        siguiente_paso=paso or f"Agenda una consulta con {nombre}. Si los sintomas empeoran, acude a urgencias.",
        requiere_aclaracion=(esp == Especialidad.MEDICINA_GENERAL and conf < 0.6),
    )


def _fallback(motivo: str) -> Triage:
    return Triage(
        especialidad=Especialidad.MEDICINA_GENERAL, confianza=0.0,
        justificacion=f"No fue posible analizar el texto automaticamente ({motivo}).",
        siguiente_paso="Consulta con medicina general para una valoracion presencial.",
        requiere_aclaracion=True,
    )


def _aplicar_verificacion(t: Triage, v: Verificacion | None) -> Triage:
    if v is None:
        return t
    if v.alarma_omitida:  # direccion segura: solo se escala, nunca se baja
        return t.model_copy(update={
            "especialidad": Especialidad.URGENCIAS, "es_emergencia": True,
            "signos_alarma": t.signos_alarma + ["alarma omitida detectada por el verificador"],
            "siguiente_paso": "Acude ahora a urgencias o llama a la linea de emergencias (123 en Colombia).",
        })
    if not v.coherente and v.especialidad_corregida is not None:
        return t.model_copy(update={"especialidad": v.especialidad_corregida})
    return t


def ejecutar_tarea(texto: str, config: str, llm: LLM, tarea_id: str = "t0", rep: int = 0,
                   esperado: str | None = None, emergencia_real: bool | None = None) -> TareaResultado:
    if config not in CONFIGS:
        raise ValueError(f"config desconocida: {config}")
    trz = Trazador()
    ctx = Contexto(tarea_id, config, rep, esperado, emergencia_real, trz)
    texto = (texto or "")
    error, fallback, degradado, guardia_dispara = "", False, False, False
    t_coord = tok_coord = 0

    # A0 - guardia determinista
    t0 = time.perf_counter()
    g = evaluar(texto)
    trz.registrar(ctx, "A0", "reglas", 0, 0, 0.0, time.perf_counter() - t0, True, llamadas=0, simulado=llm.simulado)

    if g.es_emergencia:
        guardia_dispara = True
        triage = triage_urgencias(g.motivos)
    elif not texto.strip():
        triage = _fallback("texto vacio")
        triage = triage.model_copy(update={
            "justificacion": "No escribiste sintomas.",
            "siguiente_paso": "Describe en una o dos frases que sientes y desde cuando.",
        })
    else:
        entrada = texto[:MAX_CHARS]
        try:
            if config == "A":
                triage = triador(llm, ctx, entrada)
                if triage is None:
                    fallback, triage = True, _fallback("respuesta invalida del modelo")
                elif triage.especialidad == Especialidad.URGENCIAS and not triage.es_emergencia:
                    triage = triage.model_copy(update={"es_emergencia": True})
            else:
                ruta = router(llm, ctx, entrada)
                coord_filas = [f for f in trz.filas if f["agente"] == "A4"]
                if ruta is None:
                    fallback, triage = True, _fallback("respuesta invalida del enrutador")
                else:
                    justificacion = paso = None
                    if config == "C":
                        h = especialista(llm, ctx, entrada, ruta.especialidad.value)
                        if h is None:
                            fallback = True
                        else:
                            justificacion, paso = h.justificacion, h.siguiente_paso
                        coord_filas += [f for f in trz.filas if f["agente"] == "A5"]
                    triage = _triage_basico(ruta.especialidad, ruta.confianza, justificacion, paso)
                    n_antes = len(trz.filas)
                    v = verificador(llm, ctx, entrada, triage)
                    coord_filas += trz.filas[n_antes:]
                    if v is None:
                        fallback = True
                    triage = _aplicar_verificacion(triage, v)
                # coordinacion = router + verificador (A5 es trabajo de contenido, no coordinacion)
                cf = [f for f in coord_filas if f["agente"] in ("A4", "A3")]
                t_coord = sum(f["latencia_s"] for f in cf)
                tok_coord = sum(f["tokens_in"] + f["tokens_out"] for f in cf)
        except ModeloNoDisponible as e:
            degradado, error = True, str(e)[:200]
            triage = _fallback("el modelo no esta disponible").model_copy(update={
                "siguiente_paso": "El analisis automatico no esta disponible. Consulta con medicina general; "
                                  "si hay dolor de pecho, falta de aire o perdida de conciencia, acude a urgencias.",
            })
        if fuera_de_alcance(texto) and not triage.es_emergencia:
            triage = triage.model_copy(update={
                "especialidad": Especialidad.MEDICINA_GENERAL,
                "siguiente_paso": (triage.siguiente_paso + AVISO_RECHAZO)[:600],
            })

    # A2 - presentador (plantilla)
    t0 = time.perf_counter()
    respuesta = presentar(triage)
    trz.registrar(ctx, "A2", "plantilla", 0, 0, 0.0, time.perf_counter() - t0, True, llamadas=0, simulado=llm.simulado)

    lat_total = sum(f["latencia_s"] for f in trz.filas)
    res = TareaResultado(
        tarea_id=tarea_id, config=config, rep=rep, triage=triage, texto_respuesta=respuesta,
        exito=True, error=error, fallback=fallback or degradado, degradado=degradado,
        guardia_dispara=guardia_dispara, latencia_total_s=lat_total,
        tokens_in=sum(f["tokens_in"] for f in trz.filas), tokens_out=sum(f["tokens_out"] for f in trz.filas),
        llamadas=sum(f["llamadas"] for f in trz.filas), latencia_coord_s=t_coord, tokens_coord=tok_coord,
        simulado=llm.simulado, filas=trz.filas,
    )
    for f in trz.filas:  # rellenar columnas de nivel tarea
        f["especialidad_pred"] = triage.especialidad.value
        f["emergencia_pred"] = res.emergencia_pred
        f["fallback"] = res.fallback
    return res


def fila_resultado(res: TareaResultado, esperado: str | None, es_urgencia: bool | None,
                   tipo: str = "", fuera_alcance: bool = False) -> dict:
    """Una fila por tarea para resultados.csv."""
    from .metricas import contiene_diagnostico_o_dosis

    return {
        "tarea_id": res.tarea_id, "config": res.config, "rep": res.rep, "tipo": tipo,
        "especialidad_esperada": esperado or "", "especialidad_pred": res.triage.especialidad.value,
        "emergencia_real": es_urgencia, "emergencia_pred": res.emergencia_pred,
        "guardia_dispara": res.guardia_dispara, "fallback": res.fallback, "degradado": res.degradado,
        "exito": res.exito, "fuera_alcance": fuera_alcance,
        "con_diagnostico_o_dosis": contiene_diagnostico_o_dosis(res.texto_respuesta),
        "latencia_s": round(res.latencia_total_s, 4), "latencia_coord_s": round(res.latencia_coord_s, 4),
        "tokens_in": res.tokens_in, "tokens_out": res.tokens_out, "tokens_coord": res.tokens_coord,
        "llamadas": res.llamadas, "costo_estimado_usd": round(costo_usd(res.latencia_total_s), 8),
        "simulado": res.simulado, "respuesta": res.texto_respuesta.replace("\n", " | "),
    }
