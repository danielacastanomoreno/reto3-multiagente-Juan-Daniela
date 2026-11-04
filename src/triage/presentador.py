"""A2 - Presentador (plantilla, sin LLM)."""
from __future__ import annotations

from .esquemas import Especialidad, Triage

AVISO = (
    "Esto es orientacion, no un diagnostico. Si los sintomas empeoran o aparece "
    "dolor de pecho, falta de aire o perdida de conciencia, acude a urgencias."
)

PLANTILLA_URGENCIAS = (
    "ATENCION: lo que describes puede ser una urgencia.\n"
    "Especialista recomendado: urgencias\n"
    "Motivo(s): {motivos}\n"
    "Siguiente paso: acude ahora a urgencias o llama a la linea de emergencias de tu pais (en Colombia, 123).\n"
    + AVISO
)


def triage_urgencias(motivos: list[str]) -> Triage:
    return Triage(
        especialidad=Especialidad.URGENCIAS,
        confianza=1.0,
        justificacion="Se detectaron signos de alarma que requieren valoracion inmediata.",
        signos_alarma=list(motivos),
        es_emergencia=True,
        siguiente_paso="Acude ahora a urgencias o llama a la linea de emergencias (123 en Colombia).",
    )


def presentar(t: Triage) -> str:
    if t.es_emergencia:
        return PLANTILLA_URGENCIAS.format(motivos=", ".join(t.signos_alarma) or "signos de alarma")
    alarmas = ", ".join(t.signos_alarma) if t.signos_alarma else "ninguno reportado"
    return (
        f"Especialista recomendado: {t.especialidad.value}\n"
        f"Por que: {t.justificacion}\n"
        f"Signos de alarma: {alarmas}\n"
        f"Siguiente paso: {t.siguiente_paso}\n"
        f"{AVISO}"
    )
