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

## Privacidad y observabilidad (LangSmith opcional)

LangSmith sirve para observar y depurar las corridas de `triage eval` (pasos por
agente, latencias y consumo de tokens) desde su panel. Se puede activar cuando se quiera medir o revisar una corrida:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=...
export LANGSMITH_PROJECT=reto3-triaje
uv run triage eval --configs A B --reps 2
```

Como LangSmith es un servicio SaaS, la recomendación es usarlo **solo con el dataset sintético**; si se van a probar
síntomas reales, conviene dejarlo apagado. La app Streamlit lo desactiva por defecto (`LANGSMITH_TRACING=false`)
para que el texto del paciente no salga del equipo. Las métricas del proyecto **no dependen de LangSmith**: se
calculan localmente desde los CSV.

## Cómo calculamos las métricas

`triage eval` escribe dos archivos locales:

- `data/resultados.csv`: una fila por tarea y repetición, con la especialidad esperada, la predicha, latencia total,
  tokens, llamadas, fallback, costo y respuesta generada.
- `data/trazas.csv`: una fila por paso/agente de cada tarea (A0, A1, A2, A3, A4, A5), con su latencia, tokens y TTFT.

Después, `src/triage/metricas.py` lee esos CSV y calcula las métricas **sin volver a llamar al modelo**. El
checkpoint de `--resume` evita repetir tareas ya guardadas, y tanto la app como el notebook solo leen esos resultados.

| Métrica | Cómo se calcula |
|---|---|
| Exactitud top-1 | Proporción de tareas donde `especialidad_pred == especialidad_esperada`. |
| Exactitud en ambiguos | La misma proporción, restringida a las viñetas con `tipo == "ambiguo"`. |
| Recall de urgencias | `TP / (TP + FN)` comparando `emergencia_real` con `emergencia_pred`. |
| Precisión de urgencias | `TP / (TP + FP)` con esas mismas columnas. |
| Tasa de fallback | Proporción de tareas con `fallback == True`. |
| Diagnóstico o dosis | Regex sobre la respuesta generada; se excluye el aviso fijo “no es un diagnóstico”. |
| Consistencia | Con más de una repetición: proporción de tareas cuya especialidad predicha es igual en todas las reps. |
| Latencia p50 / p95 | Percentiles 0.50 y 0.95 de `latencia_s` por tarea. |
| Tokens por tarea | Media de `tokens_in + tokens_out` por tarea. |
| Llamadas por tarea | Media de la columna `llamadas`. |
| % de latencia/tokens en coordinación | Tiempo o tokens de router (A4) + verificador (A3) dividido entre el total de la tarea. |
| Costo marginal por tarea | `latencia_s / 3600 × COSTO_HORA_EQUIPO_COP / TRM_COP_USD`; por defecto 175 COP/h y 4100 COP/USD. |
| TTFT medio | Media de `ttft_s` de las llamadas LLM (A1, A3, A4 y A5) en `trazas.csv`. |

Las metas de cada métrica están en `METAS` (`src/triage/metricas.py`) y la app marca ✅/❌ al compararlas. El juez LLM
(`llama3.1:8b`) queda como evaluación opcional de la justificación: el CLI actual no lo ejecuta y esa métrica
aparece como `n/d`.

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

### ¿La arquitectura es de un solo agente o multiagente?

El proyecto **implementa las dos**, pero la ruta que se ejecuta por defecto es de **un solo agente**.

- **Configuración A — un solo agente:** `A0 guardia → A1 triador → A2 presentador`. Solo **A1 usa el LLM**; A0 es
  un guardrail determinista y A2 una plantilla. Es la que usan por defecto la app Streamlit, la suite de robustez y
  `triage eval` si no se especifica `--configs`.
- **Configuraciones B y C — multiagente:** solo se ejecutan si se piden explícitamente con `--configs B` o
  `--configs C`.
  - **B:** router → verificador → presentador (2 llamadas al LLM).
  - **C:** router → especialista → verificador → presentador (3 llamadas).

En resumen: **el proyecto es una plataforma comparativa que tiene una línea base de un solo agente (A) y dos
variantes multiagente (B y C)**. Si se pregunta qué arquitectura corre el sistema por defecto, la respuesta es
**un solo agente (config A)**; si se pregunta qué arquitecturas evalúa el reto, son **A, B y C**.

### ¿Por qué usamos un solo agente en este proyecto?

La decisión operativa para este reto es la configuración **A** porque el **volumen de datos y el patrón de uso son
pequeños**: un dataset sintético de 30 viñetas, una app de un solo usuario, consultas secuenciales y un catálogo
cerrado de 15 especialidades. En ese escenario, una sola llamada al modelo resuelve la tarea completa; agregar
router, verificador y especialista (B/C) multiplica las llamadas y la latencia, añade puntos de fallo y no aporta
información clínica nueva. Por eso A es la línea base y la opción que corre por defecto.

En cambio, si el sistema creciera a **mucha concurrencia y alto volumen de datos**, la arquitectura **multiagente
sería la ideal**, no porque cada consulta sea más compleja, sino porque permite operar el sistema como componentes
escalables e independientes:

- **Paralelismo y escalabilidad horizontal:** el router puede repartir solicitudes entre varios especialistas y
  procesar en paralelo, mientras cada agente escala según su carga.
- **Aislamiento de fallos:** si el verificador o un especialista falla o se satura, se puede reintentar o degradar
  ese paso sin detener todo el flujo; en un solo agente, cualquier fallo afecta la respuesta completa.
- **Mantenimiento y evolución:** se pueden actualizar, versionar o reemplazar agentes por separado (por ejemplo,
  añadir RAG por especialidad o un verificador más estricto) sin reentrenar ni reescribir todo el sistema.
- **Observabilidad y control de costos:** al tener trazas por agente, es posible medir qué componente consume tiempo
  o tokens y optimizarlo; con alto volumen, esto permite asignar recursos y presupuesto por etapa.
- **Procesamiento por lotes y colas:** un orquestador multiagente encaja mejor con workers, colas y reintentos para
  absorber picos de tráfico y grandes volúmenes de solicitudes.

Eso sí: el multiagente solo entrega esas ventajas si se despliega con **orquestación, concurrencia y recursos
suficientes**. Para el tamaño actual del reto, esa infraestructura costaría más de lo que aportaría; por eso se
mantiene A como opción operativa y B/C como configuraciones experimentales para comparar.
