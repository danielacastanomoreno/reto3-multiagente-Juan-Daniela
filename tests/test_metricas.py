import pandas as pd

from triage.datos import VINETAS
from triage.metricas import contiene_diagnostico_o_dosis, cumple, metricas_config
from triage.modelos import ChatSimulado
from triage.presentador import AVISO
from triage.sistemas import ejecutar_tarea, fila_resultado


def _df(config="A", reps=1, llm=None):
    llm = llm or ChatSimulado(tasa_json_invalido=0)
    filas = []
    for rep in range(reps):
        for v in VINETAS:
            r = ejecutar_tarea(v.texto, config, llm, v.id, rep, v.esperado, v.es_urgencia)
            filas.append(fila_resultado(r, v.esperado, v.es_urgencia, v.tipo, v.fuera_alcance))
    return pd.DataFrame(filas)


def test_dataset_composicion():
    tipos = pd.Series([v.tipo for v in VINETAS]).value_counts().to_dict()
    assert tipos == {"claro": 12, "ambiguo": 6, "urgencia": 6, "fuera_alcance": 3, "ruidoso": 3}


def test_guardia_cubre_todas_las_urgencias_del_dataset():
    from triage.guardia import evaluar
    assert all(evaluar(v.texto).es_emergencia for v in VINETAS if v.es_urgencia)
    assert not any(evaluar(v.texto).es_emergencia for v in VINETAS if not v.es_urgencia)


def test_detector_diagnostico_dosis():
    assert contiene_diagnostico_o_dosis("Tienes una gastritis, toma 500 mg cada 8 horas")
    assert contiene_diagnostico_o_dosis("Toma paracetamol")
    assert not contiene_diagnostico_o_dosis(AVISO)
    assert not contiene_diagnostico_o_dosis("Especialista recomendado: urologia")


def test_metricas_en_los_tres_configs_y_recall():
    for c in ("A", "B", "C"):
        m = metricas_config(_df(c))
        assert m["recall_urgencias"] == 1.0
        assert m["tasa_diagnostico_dosis"] == 0.0
        assert m["tasa_exito"] == 1.0


def test_coordinacion_solo_en_b_y_c():
    assert metricas_config(_df("A"))["pct_lat_coord"] == 0
    assert metricas_config(_df("B"))["pct_lat_coord"] > 0
    mc = metricas_config(_df("C"))
    assert 0 < mc["pct_lat_coord"] < 1  # A5 no cuenta como coordinacion


def test_llamadas_por_config():
    for c, esperado in (("A", 1), ("B", 2), ("C", 3)):
        r = ejecutar_tarea("Me arde al orinar", c, ChatSimulado(tasa_json_invalido=0))
        assert r.llamadas == esperado


def test_urgencia_no_llama_al_modelo():
    r = ejecutar_tarea("Estoy vomitando sangre", "C", ChatSimulado())
    assert r.llamadas == 0 and r.triage.es_emergencia and r.guardia_dispara


def test_json_invalido_repara_o_cae_a_medicina_general():
    class Roto(ChatSimulado):
        def generar(self, *a, **k):
            r = super().generar(*a, **k)
            r.texto = "no soy json"
            return r

    r = ejecutar_tarea("Me arde al orinar", "A", Roto())
    assert r.fallback and r.triage.especialidad.value == "medicina_general"
    assert sum(1 for f in r.filas if f["agente"] == "A1") == 2  # intento + 1 reparacion


def test_ollama_caido_degrada():
    r = ejecutar_tarea("Me duele la rodilla", "B", ChatSimulado(caido=True))
    assert r.degradado and r.triage.especialidad.value == "medicina_general"


def test_cumple():
    assert cumple("recall_urgencias", 1.0) is True
    assert cumple("recall_urgencias", 0.9) is False


def test_tabla_metricas_para_frontend():
    from triage.metricas import CATALOGO_METRICAS, tabla_metricas
    df = pd.concat([_df("A"), _df("B")])
    t = tabla_metricas(df)
    assert len(t) == len(CATALOGO_METRICAS)
    assert {"Config A", "Config B"} <= set(t.columns)
    fila = t[t["Métrica"] == "Recall de urgencias"].iloc[0]
    assert fila["Config A"].endswith("✅")
    assert t[t["Métrica"].str.startswith("Justificación")].iloc[0]["Config A"] == "n/d"
