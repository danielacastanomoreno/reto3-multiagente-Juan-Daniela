# Informe — Sistema de triaje por especialista (reto3 · IA2)

> **Estado:** implementación completa y verificada con pruebas y con un modelo *simulado*. Las cifras empíricas
> de calidad y latencia (secciones 4–5) **están pendientes** de correr `triage eval` con Ollama en el equipo con GPU;
> no se reportan números inventados. Dataset sintético, sin validación clínica.

## 1. Problema y alcance
Entrada: texto libre (1–5 frases) con síntomas. Salida: especialidad (catálogo cerrado de 15 valores, incluidos
`urgencias` y `medicina_general`), justificación breve, signos de alarma, siguiente paso y aviso de que no es un
diagnóstico. Fuera de alcance: diagnóstico, dosis/receta, historia clínica, integración con EPS, validación clínica.

## 2. Patrón y modelos
**Router con guardrail determinista.** A0 (reglas con detección de negación, 0 tokens) decide las urgencias; si
dispara, no se llama al modelo. El resto lo resuelve un LLM local (`qwen2.5:7b-instruct-q4_K_M`, Ollama) con salida
JSON validada por Pydantic contra un Enum cerrado, un reintento de reparación y fallback a `medicina_general`.
Se comparan tres configuraciones sobre el mismo dataset: **A** (1 llamada), **B** router + verificador (2) y
**C** router + especialista + verificador (3). Descartados: orquestador–trabajadores, evaluador–optimizador en
producción y pipeline fijo (no hay subtareas independientes, borradores que iterar ni etapas de investigación).
El verificador usa el mismo 7B con otro prompt: con 8 GB de VRAM cambiar de modelo costaría 10–30 s por recarga.

## 3. Metodología
Dataset de 30 viñetas escritas antes de ver resultados (12 claras, 6 ambiguas → `medicina_general`, 6 urgencias,
3 fuera de alcance, 3 ruidosas). La especialidad se compara por igualdad exacta; el juez LLM solo evaluaría la
justificación y la ausencia de diagnóstico/dosis. Métricas: exactitud top-1 y en ambiguos; recall/precisión de
urgencias (meta recall 1.00); tasa de diagnóstico/dosis (meta 0); latencia p50/p95 y tokens; consistencia entre
repeticiones; coordinación = (t_router + t_verificador)/t_total; costo = segundos × costo/hora del equipo
(≈175 COP/h; no diferencia arquitecturas). Evaluación con checkpoint (`--resume`).

## 4. Resultados verificados sin GPU
| Verificación | Resultado |
|---|---|
| Guardrail: 22 frases de alarma, 20 con negación, 10 ambiguas | recall 1.00, 0 falsos positivos |
| Las 6 urgencias del dataset | 6/6 detectadas por A0 sin llamar al modelo; ninguna viñeta no urgente dispara |
| Esquemas, Enum cerrado, fallback por JSON inválido, degradación sin Ollama | 70 tests verdes |
| Suite de robustez (vacío, 1 palabra, 2 000 palabras, emojis, inyección, receta/dosis, Ollama apagado) | 10/10 casos OK (modelo simulado) |
| App Streamlit | alarma → banner en <1 s sin modelo; Ollama caído → medicina general con mensaje claro |

Hallazgo de diseño: en B, todo el trabajo del LLM es coordinación (no hay agente de contenido), por lo que su
`% latencia en coordinación` es ~100 % por construcción y la meta < 40 % solo es alcanzable en C.

## 5. Resultados empíricos (pendientes)
Completar tras `uv run triage eval --configs A B --reps 2` y `--configs C --reps 1` en la RTX 3050
(`data/resultados.csv`, `data/trazas.csv`); el notebook genera las tablas y figuras. Hipótesis a contrastar:
H1 A ≥ B en exactitud; H2 B no mejora el recall de urgencias; H3 B y C aumentan la latencia ≥ 40 % sin ganar calidad;
H4 más fallback en C. Si el router gana, la diferencia debería estar en el recall de urgencias, no en la especialidad.

| Config | Exactitud | Ambiguos | Recall urg. | p50 / p95 (s) | Llamadas | % coord. |
|---|---|---|---|---|---|---|
| A | _pendiente_ | | | | 1 | 0 % |
| B | _pendiente_ | | | | 2 | |
| C | _pendiente_ | | | | 3 | |

## 6. Limitaciones y trabajo futuro
Dataset pequeño y sintético; guardrail basado en reglas en español (puede fallar con redacciones no previstas);
detector de diagnóstico/dosis heurístico; consistencia con 2 repeticiones. Futuro: evaluar con el 7B y juez
`llama3.1:8b`, guía propia por especialidad (RAG) para justificar C, ampliar el dataset con revisión clínica.
