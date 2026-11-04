import pytest

from triage.guardia import evaluar

ALARMA = [
    "Dolor en el pecho que me corre al brazo izquierdo y sudo",
    "Tengo dolor de pecho y me falta el aire",
    "Siento opresión en el pecho y sudor frío",
    "Mis labios están morados y me cuesta respirar",
    "Tengo falta de aire en reposo desde hace una hora",
    "Se me desvió la cara y tengo debilidad de un lado",
    "Mi papá tiene el habla trabada de repente",
    "Tuve pérdida súbita de visión en el ojo derecho",
    "Estoy vomitando sangre",
    "Tengo heces negras desde ayer",
    "Tengo un sangrado abundante que no cede",
    "Me desmayé en el trabajo",
    "Mi hijo tuvo una convulsión",
    "Tengo fiebre, rigidez de nuca y dolor de cabeza fuerte",
    "Mi bebé de 2 meses tiene fiebre",
    "No quiero vivir más",
    "Tengo ideación suicida",
    "Me picó una abeja y tengo hinchazón de la lengua",
    "Estoy embarazada y tengo un dolor abdominal muy fuerte",
    "Me golpeé la cabeza y estoy vomitando",
    "Me quemé la cara con aceite hirviendo",
    "No tengo fiebre pero me duele el pecho y sudo mucho",
]

NEGADAS = [
    "No tengo dolor de pecho, solo tos",
    "Niego dolor de pecho y falta de aire",
    "Sin dolor de pecho ni sudoracion, solo acidez",
    "Nunca me he desmayado, pero tengo mareo leve",
    "No he tenido convulsiones",
    "No tengo rigidez de nuca, solo fiebre y dolor de cabeza leve",
    "No hay sangre en las heces, solo estreñimiento",
    "No tengo ideacion suicida, solo estoy estresado",
    "Sin fiebre en mi bebé de 2 meses, solo está inquieto",
    "No me golpeé la cabeza, me duele la rodilla",
    "Niego vomito con sangre, tengo nauseas",
    "No he perdido el conocimiento nunca",
    "No tengo labios morados, me duele la garganta",
    "Sin dificultad para respirar, tengo congestión",
    "No tengo hinchazón de la lengua, tengo picazón en la piel",
    "No estoy embarazada, tengo dolor abdominal fuerte",
    "No tengo falta de aire en reposo, solo al subir escaleras",
    "No me quemé la cara, me quemé la espalda con el sol",
    "No tengo debilidad de un lado, solo cansancio",
    "Nunca he tenido dolor en el pecho",
]

AMBIGUAS = [
    "Me siento cansado y con dolor de cabeza",
    "Tengo malestar general desde hace días",
    "Me duele un poco la barriga",
    "Tengo tos y algo de fiebre",
    "Me siento mareado a veces",
    "Tengo dolor de espalda al levantarme",
    "Estoy con mucho estrés y duermo mal",
    "Me duele la garganta y tengo congestion",
    "Tengo la piel irritada en el brazo",
    "Mi hijo tiene fiebre desde hace dos días y tos",
]


@pytest.mark.parametrize("texto", ALARMA)
def test_alarma_dispara(texto):
    r = evaluar(texto)
    assert r.es_emergencia, texto
    assert r.motivos


@pytest.mark.parametrize("texto", NEGADAS)
def test_negacion_no_dispara(texto):
    assert not evaluar(texto).es_emergencia, texto


@pytest.mark.parametrize("texto", AMBIGUAS)
def test_ambiguas_no_alarmismo(texto):
    assert not evaluar(texto).es_emergencia, texto


def test_texto_vacio():
    assert not evaluar("").es_emergencia
    assert not evaluar(None).es_emergencia


def test_recall_y_precision_global():
    tp = sum(evaluar(t).es_emergencia for t in ALARMA)
    fp = sum(evaluar(t).es_emergencia for t in NEGADAS + AMBIGUAS)
    recall = tp / len(ALARMA)
    precision = tp / (tp + fp) if tp + fp else 1.0
    assert recall == 1.0
    assert precision >= 0.90
