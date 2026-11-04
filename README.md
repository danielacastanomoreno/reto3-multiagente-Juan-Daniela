# Clasificación de Especialista según Síntomas

* **Daniela Castaño Moreno**
* **Juan José Ramos Henao**

Una persona escribe sus síntomas en una página simple y el sistema responde **a qué especialista debería acudir**.
Patrón: *router con guardrail determinista* (signos de alarma por reglas, sin LLM) + triador LLM local (Ollama).
Esto es orientación, **no un diagnóstico**. El dataset es sintético; no hay validación clínica.

## Requisitos
- Python ≥ 3.11 y [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) (solo para correr con modelo real; el resto funciona sin él)

## Instalación
```bash
uv sync --extra dev
ollama pull qwen2.5:3b                       # modelo de triaje (por defecto)
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

## Comparación: un solo agente vs. arquitectura multiagente

La configuración **A** representa el proyecto resuelto con **un solo agente de triaje** (más el guardrail
determinista A0 y la plantilla de presentación A2). Las configuraciones **B** y **C** reparten el trabajo entre
router, verificador y, en C, un especialista. La tabla resume cómo cambia el proyecto en cada caso; las cifras
empíricas de calidad y latencia son las que se obtienen con `triage eval` y quedan en `data/resultados.csv`, no
supuestos.

| Dimensión | Un solo agente (config A) | Arquitectura multiagente (configs B y C) |
|---|---|---|
| Flujo | A0 guardia → A1 triador → A2 presentador | B: A0 → A4 router → A3 verificador → A2 · C: A0 → A4 → A5 especialista → A3 → A2 |
| Llamadas al LLM | 1 | 2 (B) / 3 (C) |
| Reparto de tareas | El mismo modelo interpreta los síntomas, elige la especialidad y redacta | Se separan enrutamiento, especialización y verificación |
| Latencia esperada | Menor: una sola generación | Mayor: más llamadas y coordinación (en B, ~100 % del tiempo del LLM es coordinación) |
| Calidad sobre catálogo cerrado | Buena línea base si el modelo local respeta el esquema JSON | No garantiza mejora: el router no aporta información clínica nueva y puede introducir errores en cascada |
| Robustez | Fácil de acotar: un esquema, un fallback y un punto de fallo | Verificación cruzada y especialización, pero más puntos de fallo y más modos de degradación |
| Depuración | Sencilla: una traza por solicitud | Más compleja: hay que revisar router, verificador y especialista por separado |
| Costo y recursos | Mínimo: 1 llamada, menos tokens y menos tiempo de GPU/CPU | 2–3× llamadas y más tokens; exige más presupuesto de latencia |
| Mantenibilidad | Baja complejidad y pocas piezas | Modular y extensible (por ejemplo, RAG por especialidad), pero con más código y configuración |
| Cuándo conviene | Síntomas breves, catálogo pequeño, orientación no clínica y hardware limitado | Subtareas realmente independientes, verificación explícita o especialistas con conocimiento propio |

**Conclusión del reto:** para este problema (texto breve y catálogo cerrado de 15 especialidades) el costo de B y C
no se justifica de antemano. A es la línea base y B/C son experimentos para medir si la coordinación aporta algo;
la comparación empírica se genera en `data/resultados.csv` y se analiza en `notebooks/01_informe.ipynb`.
