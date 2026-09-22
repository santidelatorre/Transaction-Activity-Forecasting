"""Configuración mínima y reutilizable de logging."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configura un formato común para la salida de logs."""
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
