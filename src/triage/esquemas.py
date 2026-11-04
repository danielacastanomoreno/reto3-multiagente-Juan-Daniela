"""Esquemas Pydantic y catalogo cerrado de especialidades."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Especialidad(str, Enum):
    MEDICINA_GENERAL = "medicina_general"
    URGENCIAS = "urgencias"
    PEDIATRIA = "pediatria"
    GINECO_OBSTETRICIA = "gineco_obstetricia"
    CARDIOLOGIA = "cardiologia"
    NEUMOLOGIA = "neumologia"
    GASTROENTEROLOGIA = "gastroenterologia"
    NEUROLOGIA = "neurologia"
    DERMATOLOGIA = "dermatologia"
    ORTOPEDIA = "ortopedia"
    OTORRINOLARINGOLOGIA = "otorrinolaringologia"
    OFTALMOLOGIA = "oftalmologia"
    UROLOGIA = "urologia"
    PSIQUIATRIA = "psiquiatria"
    ENDOCRINOLOGIA = "endocrinologia"


class Triage(BaseModel):
    especialidad: Especialidad
    confianza: float = Field(ge=0, le=1)
    justificacion: str = Field(max_length=400)
    signos_alarma: list[str] = []
    es_emergencia: bool = False
    requiere_aclaracion: bool = False
    preguntas: list[str] = []
    siguiente_paso: str


class Ruta(BaseModel):
    especialidad: Especialidad
    confianza: float = Field(ge=0, le=1)


class Verificacion(BaseModel):
    coherente: bool
    alarma_omitida: bool
    especialidad_corregida: Especialidad | None = None
    comentarios: str = ""


class Juicio(BaseModel):
    justificacion_coherente: int = Field(ge=1, le=5)
    sin_diagnostico: bool
    sin_dosis: bool
    comentarios: str = ""


class Hallazgos(BaseModel):
    justificacion: str = Field(max_length=400)
    siguiente_paso: str
