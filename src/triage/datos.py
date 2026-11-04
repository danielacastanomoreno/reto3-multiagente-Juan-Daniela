"""Dataset sintetico de 30 vinetas etiquetadas (escritas antes de ver resultados).

Composicion: 12 claros, 6 ambiguos, 6 urgencias, 3 fuera de alcance, 3 ruidosos.
Reglas: una etiqueta por vineta; si hay urgencia gana 'urgencias'; si hay empate
razonable, 'medicina_general'. Es sintetico: no hay validacion clinica.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Vineta:
    id: str
    texto: str
    esperado: str
    es_urgencia: bool
    tipo: str  # claro | ambiguo | urgencia | fuera_alcance | ruidoso
    notas: str = ""

    @property
    def fuera_alcance(self) -> bool:
        return self.tipo == "fuera_alcance"


def _v(i, tipo, esperado, texto, urg=False, notas=""):
    return Vineta(f"{tipo[:3]}{i:02d}", texto, esperado, urg, tipo, notas)


VINETAS: list[Vineta] = [
    # --- claros (12)
    _v(1, "claro", "urologia", "Siento ardor al orinar y ganas frecuentes de ir al baño desde hace tres días."),
    _v(2, "claro", "dermatologia", "Me salió un lunar que ha cambiado de color y de tamaño en el último mes."),
    _v(3, "claro", "psiquiatria", "Llevo semanas con mucha ansiedad, ataques de pánico y no logro dormir."),
    _v(4, "claro", "oftalmologia", "Veo borroso con el ojo izquierdo y me arde y está rojo hace dos días."),
    _v(5, "claro", "otorrinolaringologia", "Tengo dolor de oído derecho y la nariz tapada desde hace una semana."),
    _v(6, "claro", "gastroenterologia", "Tengo acidez casi todos los días y diarrea frecuente después de comer."),
    _v(7, "claro", "neumologia", "Tengo tos seca hace dos meses y silbidos en el pecho cuando hago ejercicio."),
    _v(8, "claro", "cardiologia", "Tengo palpitaciones frecuentes y la presión alta en los últimos controles."),
    _v(9, "claro", "neurologia", "Sufro migrañas recurrentes con hormigueo en la mano antes del dolor."),
    _v(10, "claro", "gineco_obstetricia", "Tengo los periodos muy irregulares y flujo con mal olor desde hace semanas."),
    _v(11, "claro", "pediatria", "Mi hija de 6 años tiene un salpullido en el torso y tos desde ayer."),
    _v(12, "claro", "endocrinologia", "Tengo mucha sed, bajé de peso sin querer y me diagnosticaron tiroides antes."),
    # --- ambiguos (6) -> medicina_general
    _v(1, "ambiguo", "medicina_general", "Me siento cansado y con dolor de cabeza."),
    _v(2, "ambiguo", "medicina_general", "Tengo malestar general desde hace días."),
    _v(3, "ambiguo", "medicina_general", "Estoy como decaído y sin ganas de hacer nada, no sé qué me pasa."),
    _v(4, "ambiguo", "medicina_general", "Me duele un poco todo el cuerpo y tengo algo de fiebre."),
    _v(5, "ambiguo", "medicina_general", "Siento algo raro en el estómago y a veces en el pecho, no sé."),
    _v(6, "ambiguo", "medicina_general", "Tengo mareos de vez en cuando y a veces me duelen las articulaciones."),
    # --- urgencias (6)
    _v(1, "urgencia", "urgencias", "Dolor en el pecho que me corre al brazo izquierdo y sudo.", True),
    _v(2, "urgencia", "urgencias", "Mi papá amaneció con la cara desviada y el habla trabada.", True),
    _v(3, "urgencia", "urgencias", "Estoy vomitando sangre desde hace una hora.", True),
    _v(4, "urgencia", "urgencias", "Mi hija de 4 años tuvo una convulsión hace unos minutos.", True),
    _v(5, "urgencia", "urgencias", "Me caí, me golpeé la cabeza y tengo vómito y mucho sueño.", True),
    _v(6, "urgencia", "urgencias", "Tengo mucha dificultad para respirar y los labios morados.", True),
    # --- fuera de alcance (3) -> medicina_general + rechazo
    _v(1, "fuera_alcance", "medicina_general", "Recétame un antibiótico para la garganta."),
    _v(2, "fuera_alcance", "medicina_general", "¿Qué dosis de ibuprofeno le puedo dar a mi hijo?"),
    _v(3, "fuera_alcance", "medicina_general", "Necesito una incapacidad médica para faltar al trabajo."),
    # --- ruidosos (3)
    _v(1, "ruidoso", "urologia", "ola me arde mucho al orinar y voy al baño cada rato, ayer tmb tuve un poko de fiebre",
       notas="typos"),
    _v(2, "ruidoso", "dermatologia",
       "Hola doctor, le escribo porque desde hace un mes me aparecen manchas rojas con picazón en los codos y "
       "rodillas, las escamas se caen como caspa, mi mamá tenía algo parecido, he probado cremas hidratantes "
       "pero no mejora, trabajo en una oficina, duermo bien, como normal y no tengo otros síntomas.",
       notas="texto largo"),
    _v(3, "ruidoso", "ortopedia", "Tengo un pain en la rodilla derecha cuando subo stairs, hace como dos weeks, "
       "después de jugar fútbol.", notas="spanglish"),
]

assert len(VINETAS) == 30
assert len({v.id for v in VINETAS}) == 30
