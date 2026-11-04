"""Pagina Streamlit del triaje por especialista. Solo local; sin tracing de entradas reales."""
import os
import sys
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"  # nunca trazar sintomas reales
os.environ["LANGCHAIN_TRACING_V2"] = "false"
RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from triage.guardia import evaluar  # noqa: E402
from triage.metricas import tabla_metricas  # noqa: E402
from triage.modelos import ChatSimulado, OllamaLLM  # noqa: E402
from triage.sistemas import ejecutar_tarea  # noqa: E402

EJEMPLOS = {
    "Ardor al orinar": "Siento ardor al orinar y ganas frecuentes de ir al baño desde hace tres días.",
    "Cansancio y dolor de cabeza": "Me siento cansado y con dolor de cabeza.",
    "Dolor de pecho (alarma)": "Dolor en el pecho que me corre al brazo izquierdo y sudo.",
    "Negación (no alarma)": "No tengo dolor de pecho, solo tos seca desde hace dos meses.",
    "Pide receta": "Recétame un antibiótico para la garganta.",
}


@st.cache_resource
def cliente():
    """Cliente del modelo. TRIAGE_DRY_RUN=1 usa el simulador (demo sin GPU)."""
    if os.getenv("TRIAGE_DRY_RUN") == "1":
        return ChatSimulado()
    return OllamaLLM()


def _fuente_datos() -> Path | None:
    """data/ (real) si existe resultados.csv; si no, data/dryrun/ (simulado)."""
    for d in (RAIZ / "data", RAIZ / "data" / "dryrun"):
        if (d / "resultados.csv").exists():
            return d
    return None


@st.cache_data
def cargar(directorio: str, mtime: float):
    d = Path(directorio)
    leer = lambda n: pd.read_csv(d / n) if (d / n).exists() else None  # noqa: E731
    return leer("resultados.csv"), leer("trazas.csv"), leer("robustez.csv")


def _percentil(valores, q):
    return float(pd.Series(valores).quantile(q)) if valores else float("nan")


st.set_page_config(page_title="Triaje por especialista", page_icon="🩺", layout="wide")
st.title("🩺 Triaje por especialista")
tab_triaje, tab_metricas = st.tabs(["Orientación", "📊 Métricas"])

# ===================================================================== Orientación
with tab_triaje:
    st.warning(
        "Esta herramienta es solo orientación. **No es un diagnóstico** ni reemplaza a un profesional de la salud. "
        "En una emergencia llama al 123 o acude a urgencias."
    )
    if os.getenv("TRIAGE_DRY_RUN") == "1":
        st.info("Modo demo: respuestas simuladas, no usan un modelo real.")
    if "texto" not in st.session_state:
        st.session_state["texto"] = ""
    if "consultas" not in st.session_state:
        st.session_state["consultas"] = []  # solo metricas, nunca el texto del paciente
    with st.expander("Ejemplos"):
        for etiqueta, ejemplo in EJEMPLOS.items():
            if st.button(etiqueta):
                st.session_state["texto"] = ejemplo

    texto = st.text_area("Describe tus síntomas (1–5 frases)", key="texto", height=140, max_chars=2000)

    if st.button("Orientarme", type="primary"):
        if not texto.strip():
            st.info("Escribe qué sientes y desde cuándo.")
        else:
            alarma = evaluar(texto)
            if alarma.es_emergencia:
                st.error("🚨 **Posible urgencia.** Acude ahora a urgencias o llama al 123.")
            with st.spinner("Analizando…"):
                res = ejecutar_tarea(texto, "A", cliente(), tarea_id="app")
            t = res.triage
            st.session_state["consultas"].append({
                "latencia_s": res.latencia_total_s, "tokens": res.tokens_in + res.tokens_out,
                "llamadas": res.llamadas, "alarma": res.guardia_dispara, "fallback": res.fallback,
                "especialidad": t.especialidad.value,
            })
            if res.degradado:
                st.warning("El análisis automático no está disponible en este momento. "
                           "Te orientamos a medicina general.")
            with st.container(border=True):
                st.subheader(f"Especialista recomendado: {t.especialidad.value.replace('_', ' ')}")
                if not res.guardia_dispara and not res.degradado:
                    st.progress(float(t.confianza), text=f"Confianza: {t.confianza:.0%}")
                st.markdown(f"**Por qué:** {t.justificacion}")
                st.markdown("**Signos de alarma:** " + (", ".join(t.signos_alarma) or "ninguno reportado"))
                st.markdown(f"**Siguiente paso:** {t.siguiente_paso}")
                if t.requiere_aclaracion:
                    st.caption("La información no alcanza para afinar; si puedes, añade más detalles.")
            st.caption(f"⏱ {res.latencia_total_s:.2f} s · {res.llamadas} llamada(s) al modelo · "
                       f"{res.tokens_in + res.tokens_out} tokens")

    st.divider()
    st.caption("Esto es orientación, no un diagnóstico. Si los síntomas empeoran o aparece dolor de pecho, "
               "falta de aire o pérdida de conciencia, acude a urgencias. Proyecto académico con datos sintéticos.")

# ======================================================================= Métricas
with tab_metricas:
    # ---- sesión en vivo
    st.subheader("Esta sesión")
    cs = st.session_state.get("consultas", [])
    if not cs:
        st.caption("Aún no hay consultas en esta sesión. Haz una en la pestaña Orientación.")
    else:
        lat = [c["latencia_s"] for c in cs]
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Consultas", len(cs))
        m2.metric("Latencia p50", f"{_percentil(lat, .5):.2f} s")
        m3.metric("Latencia p95", f"{_percentil(lat, .95):.2f} s")
        m4.metric("Tokens por consulta", f"{sum(c['tokens'] for c in cs) / len(cs):.0f}")
        m5.metric("Alarmas / fallbacks", f"{sum(c['alarma'] for c in cs)} / {sum(c['fallback'] for c in cs)}")
        st.caption("Solo se guardan métricas de la sesión, nunca el texto escrito.")

    # ---- evaluación A/B/C
    st.subheader("Evaluación A/B/C")
    d = _fuente_datos()
    if d is None:
        st.info("Aún no hay resultados. Genera datos con `uv run triage eval --configs A B --resume` "
                "(o `--dry-run` para probar sin modelo) y recarga esta página.")
    else:
        df, trazas, rob = cargar(str(d), (d / "resultados.csv").stat().st_mtime)
        if bool(df["simulado"].astype(bool).any()):
            st.warning("⚠️ **Datos simulados** (`--dry-run`): validan el pipeline pero no son evidencia. "
                       "Latencias y aciertos son sintéticos.")
        else:
            st.success(f"Resultados reales · {len(df)} tareas · fuente: `{d.relative_to(RAIZ)}`")

        tabla = tabla_metricas(df, trazas, rob)
        st.markdown("**Métricas del plan vs. metas** (✅ cumple · ❌ no cumple · n/d no calculable con estos datos)")
        st.dataframe(tabla, hide_index=True, width="stretch")
        st.download_button("Descargar tabla (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                           "metricas.csv", "text/csv")
        st.caption("«Justificación (juez)» queda n/d: el juez LLM aún no está conectado al CLI. "
                   "En B, la coordinación es ~100 % por construcción (no hay agente de contenido).")

        resumen_cfg = df.groupby("config").agg(
            p50=("latencia_s", lambda s: s.quantile(.5)), p95=("latencia_s", lambda s: s.quantile(.95)),
            exactitud=("especialidad_pred", "size"),
        )
        resumen_cfg["exactitud"] = df.assign(ok=df.especialidad_pred == df.especialidad_esperada) \
            .groupby("config")["ok"].mean()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Latencia p50 / p95 (s)**")
            st.bar_chart(resumen_cfg[["p50", "p95"]])
        with c2:
            st.markdown("**Coordinación (% del total)**")
            coord = df.groupby("config").apply(
                lambda g: pd.Series({
                    "% latencia": g.latencia_coord_s.sum() / g.latencia_s.sum() if g.latencia_s.sum() else 0,
                    "% tokens": g.tokens_coord.sum() / (g.tokens_in + g.tokens_out).sum()
                    if (g.tokens_in + g.tokens_out).sum() else 0,
                }), include_groups=False)
            st.bar_chart(coord)

        c3, c4 = st.columns(2)
        with c3:
            if trazas is not None:
                st.markdown("**Latencia media por nodo (s)**")
                nodo = trazas.groupby(["config", "agente"]).latencia_s.mean().unstack(fill_value=0)
                st.bar_chart(nodo)
        with c4:
            st.markdown("**Pareto: latencia vs. exactitud**")
            fig, ax = plt.subplots(figsize=(4.5, 3.2))
            for c, fila in resumen_cfg.iterrows():
                ax.scatter(fila.p50, fila.exactitud, s=90)
                ax.annotate(f"Config {c}", (fila.p50, fila.exactitud), textcoords="offset points", xytext=(6, 4))
            ax.set_xlabel("latencia p50 (s)")
            ax.set_ylabel("exactitud top-1")
            ax.grid(alpha=.3)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**Matriz de confusión**")
        cfg = st.selectbox("Configuración", sorted(df.config.unique()))
        g = df[df.config == cfg]
        ct = pd.crosstab(g.especialidad_esperada, g.especialidad_pred)
        labs = sorted(set(ct.index) | set(ct.columns))
        ct = ct.reindex(index=labs, columns=labs, fill_value=0)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        ax.imshow(ct.values, cmap="Blues")
        ax.set_xticks(range(len(labs)), labs, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labs)), labs, fontsize=7)
        ax.set_xlabel("predicho")
        ax.set_ylabel("esperado")
        for i in range(len(labs)):
            for j in range(len(labs)):
                if ct.values[i, j]:
                    ax.text(j, i, ct.values[i, j], ha="center", va="center", fontsize=7)
        st.pyplot(fig)
        plt.close(fig)

        if rob is not None:
            st.markdown("**Suite de robustez**")
            st.dataframe(rob, hide_index=True, width="stretch")
