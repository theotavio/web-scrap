import threading
import urllib.robotparser
from typing import Dict
from urllib.parse import urlparse
import requests
from src.configuracao import obter_configuracao
from src.registro import obter_logger


class VerificadorRobots:
    _instancia = None
    _lock = threading.Lock()
    _parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def __new__(cls):
        with cls._lock:
            if cls._instancia is None:
                cls._instancia = super(VerificadorRobots, cls).__new__(cls)
            return cls._instancia

    def __init__(self):
        self.config = obter_configuracao()
        self.logger = obter_logger("robos")
        self.user_agent = self.config.obter(
            "rede.user_agent_academico",
            "ProjetoPesquisaJornalismoUFC/1.0"
        )
        self.ativo = self.config.obter("rede.respeitar_robots", True)

    def _obter_parser(self, url: str) -> urllib.robotparser.RobotFileParser:
        parsed = urlparse(url)
        dominio = parsed.netloc
        with self._lock:
            if dominio in self._parsers:
                return self._parsers[dominio]

        rp = urllib.robotparser.RobotFileParser()
        robots_url = f"{parsed.scheme}://{dominio}/robots.txt"
        rp.set_url(robots_url)

        try:
            headers = {"User-Agent": self.user_agent}
            resp = requests.get(robots_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                rp.parse(resp.text.splitlines())
            else:
                rp.allow_all = True
        except Exception:
            rp.allow_all = True

        with self._lock:
            self._parsers[dominio] = rp
        return rp

    def pode_coletar(self, url: str) -> bool:
        if not self.ativo:
            return True
        try:
            rp = self._obter_parser(url)
            return rp.can_fetch(self.user_agent, url) or rp.can_fetch("*", url)
        except Exception:
            return True
