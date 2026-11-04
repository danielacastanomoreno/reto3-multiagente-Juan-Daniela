"""Catalogo comprimido de especialidades (<= ~800 tokens) y utilidades de texto."""
from __future__ import annotations

import re
import unicodedata

from .esquemas import Especialidad

ESPECIALIDADES: dict[str, dict[str, str]] = {
    "medicina_general": {
        "sintomas_clave": "sintomas vagos, cansancio, malestar general, varios sistemas",
        "cuando_derivar": "no hay un sistema claramente afectado o falta informacion",
        "no_es_para": "urgencias claras o problemas muy especificos de un organo",
    },
    "urgencias": {
        "sintomas_clave": "dolor de pecho con irradiacion, falta de aire, convulsion, sangrado",
        "cuando_derivar": "cualquier signo de alarma",
        "no_es_para": "molestias leves o cronicas estables",
    },
    "pediatria": {
        "sintomas_clave": "nino o bebe con fiebre, tos, diarrea, erupciones",
        "cuando_derivar": "paciente menor de 14 anos",
        "no_es_para": "adultos",
    },
    "gineco_obstetricia": {
        "sintomas_clave": "dolor pelvico, sangrado menstrual anormal, embarazo, flujo",
        "cuando_derivar": "problemas del aparato reproductor femenino o embarazo",
        "no_es_para": "sintomas urinarios masculinos",
    },
    "cardiologia": {
        "sintomas_clave": "palpitaciones, presion alta, dolor de pecho leve al esfuerzo",
        "cuando_derivar": "sintomas cardiacos sin alarma",
        "no_es_para": "dolor de pecho con irradiacion o sudoracion (urgencias)",
    },
    "neumologia": {
        "sintomas_clave": "tos cronica, silbidos, falta de aire leve, asma",
        "cuando_derivar": "problemas respiratorios persistentes",
        "no_es_para": "congestion nasal o dolor de garganta (otorrino)",
    },
    "gastroenterologia": {
        "sintomas_clave": "acidez, dolor abdominal, diarrea, estrenimiento, nauseas",
        "cuando_derivar": "problemas digestivos persistentes",
        "no_es_para": "vomito con sangre (urgencias)",
    },
    "neurologia": {
        "sintomas_clave": "migrana, hormigueo, temblor, mareo recurrente, perdida de memoria",
        "cuando_derivar": "cefaleas recurrentes o sintomas neurologicos no agudos",
        "no_es_para": "debilidad subita de un lado (urgencias)",
    },
    "dermatologia": {
        "sintomas_clave": "manchas, picazon, acne, lunares, caida de pelo",
        "cuando_derivar": "problemas de piel, pelo, unas",
        "no_es_para": "quemaduras extensas (urgencias)",
    },
    "ortopedia": {
        "sintomas_clave": "dolor de rodilla, espalda, hombro, esguince, fractura leve",
        "cuando_derivar": "huesos, articulaciones, musculos",
        "no_es_para": "dolor muscular con fiebre (medicina general)",
    },
    "otorrinolaringologia": {
        "sintomas_clave": "dolor de oido, congestion, sinusitis, ronquera, amigdalas",
        "cuando_derivar": "oido, nariz, garganta",
        "no_es_para": "tos cronica con falta de aire (neumologia)",
    },
    "oftalmologia": {
        "sintomas_clave": "vision borrosa, ojo rojo, ardor ocular, ver luces",
        "cuando_derivar": "problemas de los ojos",
        "no_es_para": "perdida subita de vision (urgencias)",
    },
    "urologia": {
        "sintomas_clave": "ardor al orinar, ganas frecuentes, sangre en orina, calculos",
        "cuando_derivar": "vias urinarias, prostata, aparato reproductor masculino",
        "no_es_para": "dolor pelvico femenino (gineco)",
    },
    "psiquiatria": {
        "sintomas_clave": "ansiedad, tristeza persistente, insomnio, panico",
        "cuando_derivar": "salud mental sin riesgo inmediato",
        "no_es_para": "ideacion suicida (urgencias)",
    },
    "endocrinologia": {
        "sintomas_clave": "sed excesiva, tiroides, aumento o perdida de peso inexplicable",
        "cuando_derivar": "diabetes, tiroides, trastornos hormonales",
        "no_es_para": "obesidad sin causa hormonal (medicina general)",
    },
}

assert set(ESPECIALIDADES) == {e.value for e in Especialidad}


def catalogo_comprimido() -> str:
    lineas = []
    for nombre, d in ESPECIALIDADES.items():
        lineas.append(
            f"- {nombre}: {d['sintomas_clave']}. Derivar si {d['cuando_derivar']}. "
            f"No para {d['no_es_para']}."
        )
    return "\n".join(lineas)


def normalizar(texto: str) -> str:
    """Minusculas, sin tildes, espacios colapsados."""
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t).strip()
