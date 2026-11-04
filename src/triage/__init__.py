"""Sistema de triaje por especialista: router con guardrail determinista."""
import os
from pathlib import Path

__version__ = "0.1.0"


def _cargar_env() -> None:
    """Carga <raiz>/.env (KEY=VALOR) sin pisar variables ya definidas en el entorno."""
    ruta = Path(__file__).resolve().parents[2] / ".env"
    if not ruta.is_file():
        return
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#") or "=" not in linea:
            continue
        clave, valor = linea.split("=", 1)
        valor = valor.split(" #")[0].strip().strip("'\"")
        if clave.strip() and valor:
            os.environ.setdefault(clave.strip(), valor)


_cargar_env()
