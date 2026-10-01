from pathlib import Path
from typing import Any, Dict
import yaml


class Configuracao:
    _instancia = None
    _dados: Dict[str, Any] = {}

    def __new__(cls, caminho_config: str = "config.yaml"):
        if cls._instancia is None:
            cls._instancia = super(Configuracao, cls).__new__(cls)
            cls._instancia._carregar(caminho_config)
        return cls._instancia

    def _carregar(self, caminho_config: str) -> None:
        caminho = Path(caminho_config)
        if not caminho.exists():
            caminho = Path(__file__).resolve().parent.parent / caminho_config
        with open(caminho, "r", encoding="utf-8") as f:
            self._dados = yaml.safe_load(f)

    def obter(self, chave: str, padrao: Any = None) -> Any:
        chaves = chave.split(".")
        valor = self._dados
        for k in chaves:
            if isinstance(valor, dict) and k in valor:
                valor = valor[k]
            else:
                return padrao
        return valor

    @property
    def dados(self) -> Dict[str, Any]:
        return self._dados


def obter_configuracao(caminho_config: str = "config.yaml") -> Configuracao:
    return Configuracao(caminho_config)
