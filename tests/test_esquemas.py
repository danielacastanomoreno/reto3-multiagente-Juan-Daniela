import pytest
from pydantic import ValidationError

from triage.catalogo import ESPECIALIDADES, catalogo_comprimido
from triage.esquemas import Especialidad, Ruta, Triage
from triage.presentador import presentar, triage_urgencias


def test_catalogo_coincide_con_enum():
    assert set(ESPECIALIDADES) == {e.value for e in Especialidad}
    assert len(Especialidad) == 15


def test_catalogo_presupuesto_tokens():
    # aprox. 1 token ~ 4 caracteres; presupuesto <= 800 tokens
    assert len(catalogo_comprimido()) / 4 <= 800


def test_enum_cerrado():
    with pytest.raises(ValidationError):
        Ruta(especialidad="cardiologo_estrella", confianza=0.5)
    with pytest.raises(ValidationError):
        Triage(especialidad="inventada", confianza=0.5, justificacion="x", siguiente_paso="y")


def test_confianza_en_rango():
    with pytest.raises(ValidationError):
        Ruta(especialidad="urologia", confianza=1.5)


def test_presentador_incluye_aviso():
    t = Triage(
        especialidad=Especialidad.UROLOGIA, confianza=0.9,
        justificacion="Sintomas urinarios.", siguiente_paso="Agenda cita.",
    )
    txt = presentar(t)
    assert "urologia" in txt and "no un diagnostico" in txt
    assert "ninguno reportado" in txt


def test_presentador_urgencias():
    txt = presentar(triage_urgencias(["posible ACV"]))
    assert "urgencias" in txt and "posible ACV" in txt
