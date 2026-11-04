# Triaje por especialista (reto3 · IA2)

Una persona escribe sus síntomas en una página simple y el sistema responde **a qué especialista debería acudir**.
Patrón: *router con guardrail determinista* (signos de alarma por reglas, sin LLM) + triador LLM local (Ollama).
Esto es orientación, **no un diagnóstico**. El dataset es sintético; no hay validación clínica.

## Requisitos
- Python ≥ 3.11 y [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) (solo para correr con modelo real; el resto funciona sin él)

## Instalación
```bash
uv sync --extra dev
ollama pull qwen2.5:7b-instruct-q4_K_M      # GPU de 8 GB
# sin GPU / 4–6 GB:  ollama pull qwen2.5:3b-instruct-q4_K_M  y  export TRIAGE_MODEL=qwen2.5:3b-instruct-q4_K_M
ollama pull llama3.1:8b-instruct-q4_K_M      # juez offline (opcional, al final)
cp .env.example .env
```
Variables recomendadas de Ollama: `OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=30m`.

## Uso
```bash
uv run triage doctor                                   # hardware, modelo, warm-up y esquema válido
uv run pytest -q                                       # guardrail, esquemas, métricas
uv run triage eval --configs A --limit 5               # corrida de humo (modelo real)
uv run triage eval --configs A B --reps 2 --resume     # evaluación (checkpoint por tarea)
uv run triage eval --configs C --reps 1 --resume       # C una sola repetición (por costo)
uv run triage robustez                                 # suite de robustez
uv run triage serve                                    # app Streamlit  (= streamlit run app/app.py)
```
**Sin GPU ni Ollama** (probar el pipeline): `uv run triage eval --dry-run --configs A B C --reps 2 --reps-c 1` y
`uv run triage robustez --dry-run`. Escribe en `data/dryrun/` con un `ChatSimulado`: sus latencias y aciertos
son **sintéticos** y el notebook los marca con una advertencia. Para la demo de la app sin modelo: `TRIAGE_DRY_RUN=1 uv run triage serve`.

El notebook `notebooks/01_informe.ipynb` lee `data/resultados.csv` (o `data/dryrun/` si aún no existe) y corre en segundos;
`REPETIR = False` por defecto.

## Configuraciones
| Config | Ruta | Llamadas LLM |
|---|---|---|
| A | A0 guardia → A1 triador → A2 presentador | 1 |
| B | A0 → A4 router → A3 verificador → A2 | 2 |
| C | A0 → A4 → A5 especialista → A3 → A2 | 3 |

Si la guardia detecta una alarma, se responde con la plantilla de urgencias y **no se llama al modelo**.
Si Ollama no responde, A0+A2 siguen funcionando y el resto degrada a medicina general con un mensaje claro.

## Estructura
```
src/triage/   catalogo, esquemas, guardia (A0), prompts, modelos, agentes, sistemas (A/B/C), metricas, datos, robustez, cli
app/app.py    página Streamlit
notebooks/    01_informe.ipynb
tests/        test_guardia, test_esquemas, test_metricas
data/         resultados.csv, trazas.csv (reales) · dryrun/ (simulados, ignorado por git)
docs/         informe.md
```

## Privacidad
LangSmith es opcional y SaaS: úsalo solo con el dataset sintético (`LANGSMITH_TRACING=true` en tu shell al correr `eval`).
La app fuerza `LANGSMITH_TRACING=false`. No se guardan síntomas reales.
