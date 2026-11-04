"""CLI: triage eval | doctor | robustez | serve."""
from __future__ import annotations

import argparse
import csv
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .datos import VINETAS
from .metricas import cumple, resumen
from .modelos import BASE_URL, MODELO_DEFECTO, OllamaLLM, ChatSimulado, crear_llm, healthcheck
from .sistemas import CONFIGS, ejecutar_tarea, fila_resultado

RAIZ = Path(__file__).resolve().parents[2]


def _leer_hechos(ruta: Path) -> set[tuple[str, int, str]]:
    if not ruta.exists():
        return set()
    with ruta.open(encoding="utf-8") as f:
        return {(r["config"], int(r["rep"]), r["tarea_id"]) for r in csv.DictReader(f)}


def _anexar(ruta: Path, filas: list[dict]) -> None:
    if not filas:
        return
    nuevo = not ruta.exists() or ruta.stat().st_size == 0
    with ruta.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0].keys()))
        if nuevo:
            w.writeheader()
        w.writerows(filas)


def cmd_eval(a: argparse.Namespace) -> int:
    out = Path(a.out) if a.out else RAIZ / "data" / ("dryrun/resultados.csv" if a.dry_run else "resultados.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    trazas = out.with_name("trazas.csv")
    if not a.resume:
        out.unlink(missing_ok=True)
        trazas.unlink(missing_ok=True)
    hechos = _leer_hechos(out)
    if a.dry_run:
        llm = ChatSimulado()
        print("[dry-run] ChatSimulado: latencias y aciertos SINTETICOS, no son evidencia.")
    else:
        ok, det = healthcheck()
        if not ok:
            print(f"ERROR: {det}. Usa --dry-run para probar el pipeline sin modelo.", file=sys.stderr)
            return 1
        llm = crear_llm("triador")
    vinetas = VINETAS[: a.limit] if a.limit else VINETAS
    if a.limit:  # humo representativo: toma un poco de cada tipo
        por_tipo: dict[str, list] = {}
        for v in VINETAS:
            por_tipo.setdefault(v.tipo, []).append(v)
        vinetas = []
        while len(vinetas) < a.limit and any(por_tipo.values()):
            for lista in por_tipo.values():
                if lista and len(vinetas) < a.limit:
                    vinetas.append(lista.pop(0))
    total = len(a.configs) * a.reps * len(vinetas)
    hecho = 0
    for config in a.configs:
        reps = a.reps_c if (config == "C" and a.reps_c is not None) else a.reps
        for rep in range(reps):
            for v in vinetas:
                if (config, rep, v.id) in hechos:
                    continue
                res = ejecutar_tarea(v.texto, config, llm, v.id, rep, v.esperado, v.es_urgencia)
                _anexar(out, [fila_resultado(res, v.esperado, v.es_urgencia, v.tipo, v.fuera_alcance)])
                _anexar(trazas, res.filas)  # checkpoint: una tarea por escritura
                hecho += 1
                print(f"\r[{hecho}/{total}] {config} rep{rep} {v.id}", end="", flush=True)
    print(f"\nListo: {out}")
    if out.exists():
        import pandas as pd

        print(resumen(pd.read_csv(out)).round(3).T.to_string())
    return 0


def _hardware() -> tuple[str, str]:
    if shutil.which("nvidia-smi"):
        try:
            s = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip()
            mib = int(s.split(",")[-1].strip().split()[0])
            perfil = "GPU8" if mib >= 7500 else "GPU4"
            return perfil, f"GPU: {s}"
        except Exception:
            pass
    return "CPU", "sin GPU NVIDIA detectada (nvidia-smi ausente)"


def cmd_doctor(_: argparse.Namespace) -> int:
    perfil, det = _hardware()
    print(f"Python {platform.python_version()} | {det} | perfil: {perfil}")
    if perfil != "GPU8":
        print("  -> mantener qwen2.5:3b (TRIAGE_MODEL) y descartar la config C")
    ok, det = healthcheck()
    print(("OK   " if ok else "FALLO ") + det)
    if not ok:
        return 1
    from .agentes import Contexto, Trazador, triador

    llm = OllamaLLM(MODELO_DEFECTO)
    ctx = Contexto("doctor", "A", 0, None, None, Trazador())
    t = triador(llm, ctx, "Me arde al orinar y voy mucho al baño")  # warm-up + validacion de esquema
    f = ctx.trazador.filas
    print(f"warm-up/primera llamada: {f[0]['latencia_s']:.1f}s, tokens {f[0]['tokens_in']}+{f[0]['tokens_out']}")
    print("esquema valido:", t is not None, "->", t.especialidad.value if t else "n/a")
    return 0 if t else 1


def cmd_robustez(a: argparse.Namespace) -> int:
    from .robustez import correr

    return correr(Path(a.out) if a.out else RAIZ / "data" / "robustez.csv", a.dry_run)


def cmd_serve(_: argparse.Namespace) -> int:
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(RAIZ / "app" / "app.py")])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="triage", description="Triaje por especialista (reto3)")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="evaluar configuraciones sobre el dataset")
    e.add_argument("--configs", nargs="+", default=["A"], choices=CONFIGS)
    e.add_argument("--reps", type=int, default=1)
    e.add_argument("--reps-c", type=int, default=None, help="repeticiones solo para C (p.ej. 1)")
    e.add_argument("--limit", type=int, default=0, help="humo: solo N vinetas")
    e.add_argument("--out", default="")
    e.add_argument("--resume", action="store_true")
    e.add_argument("--dry-run", action="store_true", help="ChatSimulado, sin Ollama")
    e.set_defaults(fn=cmd_eval)
    d = sub.add_parser("doctor", help="verifica modelo, hardware y esquema")
    d.set_defaults(fn=cmd_doctor)
    r = sub.add_parser("robustez", help="suite de robustez (aparte de la exactitud)")
    r.add_argument("--out", default="")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(fn=cmd_robustez)
    s = sub.add_parser("serve", help="lanza la app Streamlit")
    s.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
