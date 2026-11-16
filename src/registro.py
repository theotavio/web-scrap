import logging
import time
from typing import Any, Callable, Dict, List, Optional


MEMORIA_LOGS: List[Dict[str, Any]] = []
OUVINTES_LOGS: List[Callable[[str, str, str, str], None]] = []


class BufferLogHandler(logging.Handler):
    def __init__(self, nome_etapa: str):
        super().__init__()
        self.nome_etapa = nome_etapa

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            hora = time.strftime("%H:%M:%S")
            item = {
                "etapa": self.nome_etapa,
                "nivel": record.levelname,
                "mensagem": msg,
                "hora": hora
            }
            MEMORIA_LOGS.append(item)
            if len(MEMORIA_LOGS) > 2000:
                MEMORIA_LOGS.pop(0)

            for ouvinte in list(OUVINTES_LOGS):
                try:
                    ouvinte(self.nome_etapa, msg, record.levelname, hora)
                except Exception:
                    pass
        except Exception:
            self.handleError(record)


def obter_logger(nome_etapa: str) -> logging.Logger:
    logger = logging.getLogger(f"jornalismo.{nome_etapa}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S"
    )

    handler = BufferLogHandler(nome_etapa)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formato)
    logger.addHandler(handler)

    return logger


def adicionar_ouvinte_logs(callback: Callable[[str, str, str, str], None]) -> None:
    if callback not in OUVINTES_LOGS:
        OUVINTES_LOGS.append(callback)


def remover_ouvinte_logs(callback: Callable[[str, str, str, str], None]) -> None:
    if callback in OUVINTES_LOGS:
        OUVINTES_LOGS.remove(callback)


def limpar_logs() -> None:
    MEMORIA_LOGS.clear()


def obter_logs(etapa: Optional[str] = None) -> List[Dict[str, Any]]:
    if not etapa or etapa == "todas":
        return list(MEMORIA_LOGS)
    return [item for item in MEMORIA_LOGS if item["etapa"] == etapa]
