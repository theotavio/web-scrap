import logging
from pathlib import Path
from src.configuracao import obter_configuracao


def obter_logger(nome_etapa: str) -> logging.Logger:
    config = obter_configuracao()
    dir_logs = Path(config.obter("geral.diretorio_logs", "logs"))
    dir_logs.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"jornalismo.{nome_etapa}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False

    formato = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    arquivo_etapa = dir_logs / f"{nome_etapa}.log"
    handler_etapa = logging.FileHandler(arquivo_etapa, encoding="utf-8")
    handler_etapa.setLevel(logging.INFO)
    handler_etapa.setFormatter(formato)
    logger.addHandler(handler_etapa)

    arquivo_erros = dir_logs / "erros.log"
    handler_erros = logging.FileHandler(arquivo_erros, encoding="utf-8")
    handler_erros.setLevel(logging.ERROR)
    handler_erros.setFormatter(formato)
    logger.addHandler(handler_erros)

    return logger


def limpar_logs() -> None:
    config = obter_configuracao()
    dir_logs = Path(config.obter("geral.diretorio_logs", "logs"))
    if dir_logs.exists():
        for arq in dir_logs.glob("*.log"):
            try:
                with open(arq, "w", encoding="utf-8") as f:
                    f.truncate(0)
            except Exception:
                pass
