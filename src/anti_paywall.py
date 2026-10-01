import json
import re
from typing import Any, Dict, Optional, Tuple
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.configuracao import obter_configuracao
from src.registro import obter_logger


class BypassCascataPaywall:
    def __init__(self):
        self.config = obter_configuracao()
        self.logger = obter_logger("paywall")
        self.timeout = self.config.obter("rede.timeout_requisicao", 15)
        self.ua_academico = self.config.obter(
            "rede.user_agent_academico",
            "ProjetoPesquisaJornalismoUFC/1.0"
        )
        self.ua_googlebot = self.config.obter(
            "rede.user_agent_googlebot",
            "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
        )
        self.sessao = self._criar_sessao()

    def _criar_sessao(self) -> requests.Session:
        sessao = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        sessao.mount("http://", adapter)
        sessao.mount("https://", adapter)
        return sessao

    def _obter_headers_bypass(self, modo: str = "googlebot") -> Dict[str, str]:
        if modo == "googlebot":
            return {
                "User-Agent": self.ua_googlebot,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
                "Referer": "https://www.google.com/",
                "X-Forwarded-For": "66.249.66.1",
            }
        elif modo == "academico":
            return {
                "User-Agent": self.ua_academico,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
            }
        else:
            return {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "pt-BR,pt;q=0.9",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "cross-site",
                "Referer": "https://t.co/",
            }

    def _extrair_json_ld(self, html: str) -> Optional[Tuple[str, str, Optional[str], Optional[str]]]:
        try:
            soup = BeautifulSoup(html, "html.parser")
            scripts = soup.find_all("script", type="application/ld+json")
            for script in scripts:
                if not script.string:
                    continue
                try:
                    dado = json.loads(script.string.strip())
                    if isinstance(dado, list):
                        itens = dado
                    elif isinstance(dado, dict) and "@graph" in dado:
                        itens = dado["@graph"]
                    else:
                        itens = [dado]

                    for item in itens:
                        if not isinstance(item, dict):
                            continue
                        tipo = item.get("@type", "")
                        if tipo in ("NewsArticle", "ReportageNewsArticle", "Article", "BlogPosting"):
                            titulo = item.get("headline") or item.get("name")
                            corpo = item.get("articleBody") or item.get("description")
                            data_pub = item.get("datePublished") or item.get("dateCreated")
                            autor = None
                            if "author" in item:
                                if isinstance(item["author"], dict):
                                    autor = item["author"].get("name")
                                elif isinstance(item["author"], list) and item["author"]:
                                    autor = item["author"][0].get("name") if isinstance(item["author"][0], dict) else str(item["author"][0])
                                elif isinstance(item["author"], str):
                                    autor = item["author"]

                            if titulo and corpo and len(corpo.strip()) > 150:
                                return (titulo.strip(), corpo.strip(), data_pub, autor)
                except Exception:
                    continue
        except Exception:
            pass
        return None

    def baixar_conteudo(self, url: str, timestamp_cdx: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        metadados: Dict[str, Any] = {"metodo": "desconhecido", "tamanho": 0}

        if timestamp_cdx:
            url_wayback = f"http://web.archive.org/web/{timestamp_cdx}id_/{url}"
            try:
                r = self.sessao.get(url_wayback, headers=self._obter_headers_bypass("academico"), timeout=self.timeout)
                if r.status_code == 200 and len(r.text) > 500:
                    metadados["metodo"] = "web_archive_cdx"
                    metadados["tamanho"] = len(r.text)
                    return r.text, "wayback", metadados
            except Exception:
                pass

        try:
            r = self.sessao.get(url, headers=self._obter_headers_bypass("googlebot"), timeout=self.timeout)
            if r.status_code == 200 and len(r.text) > 500:
                json_ld = self._extrair_json_ld(r.text)
                if json_ld:
                    metadados["metodo"] = "googlebot_json_ld"
                    metadados["dados_json_ld"] = json_ld
                else:
                    metadados["metodo"] = "googlebot_html"
                metadados["tamanho"] = len(r.text)
                return r.text, "googlebot", metadados
        except Exception:
            pass

        try:
            r = self.sessao.get(url, headers=self._obter_headers_bypass("navegador"), timeout=self.timeout)
            if r.status_code == 200 and len(r.text) > 500:
                metadados["metodo"] = "navegador_padrao"
                metadados["tamanho"] = len(r.text)
                return r.text, "navegador", metadados
        except Exception:
            pass

        try:
            r = self.sessao.get(url, headers=self._obter_headers_bypass("academico"), timeout=self.timeout)
            if r.status_code == 200 and len(r.text) > 500:
                metadados["metodo"] = "academico_padrao"
                metadados["tamanho"] = len(r.text)
                return r.text, "academico", metadados
        except Exception:
            pass

        self.logger.warning(f"Download nao concluido para {url}")
        return None, None, metadados
