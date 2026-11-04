"""A0 - Guardia determinista de signos de alarma (0 tokens, sin LLM).

Cada regla es una conjuncion de grupos de patrones: dispara solo si TODOS los
grupos tienen al menos una coincidencia no negada. La negacion se detecta con
una ventana de 5 tokens antes del termino, que se corta en puntuacion y en
conjunciones adversativas ("pero", "aunque"...), para no ocultar una alarma
real que aparece despues de una negacion ("no tengo fiebre pero me duele el
pecho y sudo").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalogo import normalizar

NEGADORES = {"no", "sin", "nunca", "niego", "jamas", "ni", "tampoco", "descarto"}
CORTES = {"pero", "aunque", "sino", "excepto", "salvo"}
VENTANA = 5


@dataclass(frozen=True)
class Regla:
    motivo: str
    grupos: tuple[re.Pattern, ...]


def _r(motivo: str, *patrones: str) -> Regla:
    return Regla(motivo, tuple(re.compile(p) for p in patrones))


REGLAS: tuple[Regla, ...] = (
    _r(
        "posible sindrome coronario",
        r"dolor (\w+ )?(de|en|del) (el |la )?(pecho|torax)|opresion (de|en) (el )?pecho|presion en el pecho"
        r"|(me )?duele\w* (mucho )?(el |en el )?(pecho|torax)|dolor toracico",
        r"brazo|mandibula|\bsudo\b|sudor|sudand|falta de aire|me falta el aire|ahogo",
    ),
    _r(
        "insuficiencia respiratoria",
        r"falta de aire (en|en el) reposo|labios (\w+ )?(morados|azules)|cianosis|no puedo respirar|me ahogo",
    ),
    _r(
        "posible ACV",
        r"cara (desviada|caida)|debilidad (de|en) (un|el) lado|habla (trabada|arrastrada)"
        r"|no puedo hablar bien|perdida subita de (la )?vision|perdi (de golpe )?la vision"
        r"|se me (durmio|paralizo) (la cara|un lado)",
    ),
    _r(
        "hemorragia",
        r"sangrado abundante|sangrado (que )?no (cede|para)|vomit\w* (con |de )?sangre"
        r"|heces negras|deposiciones negras|melena",
    ),
    _r(
        "evento neurologico agudo",
        r"desmay|convulsion|perdida de (la )?conciencia|perdi (el )?conocimiento|me quede inconsciente",
    ),
    _r(
        "sepsis/meningitis",
        r"fiebre|febril",
        r"rigidez de (la )?nuca|nuca rigida|cuello rigido|rigidez en el cuello",
        r"dolor de cabeza|cefalea",
    ),
    _r(
        "fiebre en menor de 3 meses",
        r"fiebre|febril|temperatura",
        r"\bbebe\b|recien nacid|neonat|lactante",
        r"\b(un|1|2|3|dos|tres)\s+mes(es)?\b|recien nacid|neonat|\b\w+\s+semanas? de (nacid|vida|edad)",
    ),
    _r(
        "riesgo psiquiatrico",
        r"ideacion suicida|suicid|autolesion|hacerme dano|quiero morir|no quiero vivir"
        r"|quitarme la vida|matarme|cortarme las venas",
    ),
    _r(
        "anafilaxia",
        r"(hinchazon|inflamacion) (de|en) (la )?(lengua|garganta)|(lengua|garganta) hinchad",
    ),
    _r(
        "anafilaxia",
        r"dificultad para respirar|me cuesta respirar",
        r"comi\b|comida|picadura|me pico|abeja|avispa|mariscos|mani\b|medicamento|inyeccion",
    ),
    _r(
        "posible embarazo ectopico",
        r"abdomen|abdominal|barriga|vientre|pelvi",
        r"fuerte|intenso|insoportable|severo",
        r"embaraz",
    ),
    _r(
        "trauma craneoencefalico",
        r"golpe (en|con)? ?(la )?cabeza|golpee (la )?cabeza|traumatismo (en la )?cabeza|me pegue en la cabeza",
        r"vomit|somnolen|mucho sueno|confus|adormil",
    ),
    _r(
        "quemadura grave",
        r"quemadura\w* (extensa|grave|grande|severa|de tercer grado)"
        r"|quemadura\w* (en|de) (la )?(cara|manos?|genitales)"
        r"|me quem\w+ (la )?(cara|manos?|genitales)",
    ),
)


@dataclass
class ResultadoGuardia:
    es_emergencia: bool
    motivos: list[str] = field(default_factory=list)


def _negado(texto: str, inicio: int) -> bool:
    previo = re.split(r"[.;,:!?\n]", texto[:inicio])[-1]
    tokens = re.findall(r"\w+", previo)
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] in CORTES:
            tokens = tokens[i + 1 :]
            break
    ventana = tokens[-VENTANA:]
    return any(t in NEGADORES for t in ventana)


def _grupo_activo(patron: re.Pattern, texto: str) -> bool:
    return any(not _negado(texto, m.start()) for m in patron.finditer(texto))


def evaluar(texto: str) -> ResultadoGuardia:
    t = normalizar(texto or "")
    motivos: list[str] = []
    for regla in REGLAS:
        if all(_grupo_activo(g, t) for g in regla.grupos):
            if regla.motivo not in motivos:
                motivos.append(regla.motivo)
    return ResultadoGuardia(es_emergencia=bool(motivos), motivos=motivos)


_FUERA_ALCANCE = re.compile(
    r"\brecet\w*|\bdosis\b|\bincapacidad\b|\bformula medica\b|\bque (medicamento|pastilla|antibiotico) (me )?(tomo|puedo tomar|debo tomar)\b"
    r"|\bcuantas? (pastillas|gotas|tabletas)\b|\bantibiotico para\b|\bresultados? de (mi|los) examen"
)


def fuera_de_alcance(texto: str) -> bool:
    """Detecta peticiones de receta, dosis, incapacidad o interpretacion de examenes."""
    return bool(_FUERA_ALCANCE.search(normalizar(texto or "")))
