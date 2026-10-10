import json
import re
import threading
import time
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
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
        self.timeout_wayback = (
            self.config.obter("rede.timeout_conexao_wayback", 8),
            self.config.obter("rede.timeout_leitura_wayback", 15),
        )
        self.intervalo_wayback = self.config.obter("rede.intervalo_wayback_segundos", 1.5)
        self.pausa_base_wayback = self.config.obter("rede.pausa_wayback_indisponivel", 120)
        self.pausa_max_wayback = self.config.obter("rede.pausa_wayback_maxima", 1800)
        self.ua_academico = self.config.obter(
            "rede.user_agent_academico",
            "ProjetoPesquisaJornalismoUFC/1.0"
        )
        self.ua_googlebot = self.config.obter(
            "rede.user_agent_googlebot",
            "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)"
        )
        self._local = threading.local()
        self._wayback_bloqueado_ate = 0.0
        self._wayback_falhas_seguidas = 0
        self._wayback_ultimo_acesso = 0.0

    def _obter_sessao(self) -> requests.Session:
        if not hasattr(self._local, "sessao"):
            sessao = requests.Session()
            retries = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                raise_on_status=False
            )
            adapter = HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=5)
            sessao.mount("http://", adapter)
            sessao.mount("https://", adapter)
            self._local.sessao = sessao
        return self._local.sessao

    def _obter_sessao_wayback(self) -> requests.Session:
        if not hasattr(self._local, "sessao_wayback"):
            sessao = requests.Session()
            retries = Retry(total=0, connect=0, read=0, status=0, redirect=5, raise_on_status=False)
            adapter = HTTPAdapter(max_retries=retries, pool_connections=2, pool_maxsize=2)
            sessao.mount("https://", adapter)
            sessao.mount("http://", adapter)
            self._local.sessao_wayback = sessao
        return self._local.sessao_wayback

    def _wayback_disponivel(self) -> bool:
        return time.monotonic() >= self._wayback_bloqueado_ate

    def _registrar_falha_wayback(self) -> None:
        self._wayback_falhas_seguidas += 1
        pausa = min(
            self.pausa_base_wayback * (2 ** (self._wayback_falhas_seguidas - 1)),
            self.pausa_max_wayback
        )
        self._wayback_bloqueado_ate = time.monotonic() + pausa
        self.logger.warning(
            f"Wayback indisponivel; pausando consultas ao Wayback por {int(pausa)}s "
            f"(falhas seguidas: {self._wayback_falhas_seguidas})"
        )

    def _registrar_sucesso_wayback(self) -> None:
        self._wayback_falhas_seguidas = 0

    def _respeitar_intervalo_wayback(self) -> None:
        espera = self._wayback_ultimo_acesso + self.intervalo_wayback - time.monotonic()
        if espera > 0:
            time.sleep(espera)
        self._wayback_ultimo_acesso = time.monotonic()

    @staticmethod
    def _normalizar_url_ao_vivo(url: str) -> str:
        partes = urlsplit(url)
        host = partes.netloc
        if host.endswith(":80") or host.endswith(":443"):
            host = host.rsplit(":", 1)[0]
        consulta = [
            (k, v) for k, v in parse_qsl(partes.query, keep_blank_values=True)
            if k.lower() not in ("amp", "ref") and not k.lower().startswith("utm_")
        ]
        esquema = "https" if partes.scheme == "http" else partes.scheme
        return urlunsplit((esquema, host, partes.path, urlencode(consulta), ""))

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

    def _baixar_wayback(self, url: str, timestamp_cdx: str) -> Optional[str]:
        if not self._wayback_disponivel():
            return None
        self._respeitar_intervalo_wayback()
        url_wayback = f"https://web.archive.org/web/{timestamp_cdx}id_/{url}"
        try:
            r = self._obter_sessao_wayback().get(
                url_wayback,
                headers=self._obter_headers_bypass("academico"),
                timeout=self.timeout_wayback
            )
        except (requests.exceptions.ConnectTimeout, requests.exceptions.ConnectionError):
            self._registrar_falha_wayback()
            return None
        except Exception:
            return None

        if r.status_code in (403, 429, 503):
            self._registrar_falha_wayback()
            return None

        self._registrar_sucesso_wayback()
        if r.status_code == 200 and len(r.text) > 500:
            return r.text
        return None

    def baixar_conteudo(self, url: str, timestamp_cdx: Optional[str] = None) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        metadados: Dict[str, Any] = {"metodo": "desconhecido", "tamanho": 0}
        sessao = self._obter_sessao()

        if timestamp_cdx:
            html_wayback = self._baixar_wayback(url, timestamp_cdx)
            if html_wayback:
                metadados["metodo"] = "web_archive_cdx"
                metadados["tamanho"] = len(html_wayback)
                return html_wayback, "wayback", metadados

        url_viva = self._normalizar_url_ao_vivo(url)

        try:
            r = sessao.get(url_viva, headers=self._obter_headers_bypass("googlebot"), timeout=self.timeout)
            if r.status_code in (404, 410):
                self.logger.warning(f"Pagina inexistente ({r.status_code}) para {url}")
                return None, None, metadados
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
            r = sessao.get(url_viva, headers=self._obter_headers_bypass("navegador"), timeout=self.timeout)
            if r.status_code in (404, 410):
                return None, None, metadados
            if r.status_code == 200 and len(r.text) > 500:
                metadados["metodo"] = "navegador_padrao"
                metadados["tamanho"] = len(r.text)
                return r.text, "navegador", metadados
        except Exception:
            pass

        try:
            r = sessao.get(url_viva, headers=self._obter_headers_bypass("academico"), timeout=self.timeout)
            if r.status_code == 200 and len(r.text) > 500:
                metadados["metodo"] = "academico_padrao"
                metadados["tamanho"] = len(r.text)
                return r.text, "academico", metadados
        except Exception:
            pass

        self.logger.warning(f"Download nao concluido para {url}")
        return None, None, metadados
