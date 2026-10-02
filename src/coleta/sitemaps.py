import gzip
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.coleta.filtros import FiltroUrls


class ColetorSitemaps:
    def __init__(self):
        self.config = obter_configuracao()
        self.logger = obter_logger("coleta")
        self.filtro = FiltroUrls()
        self.user_agent = self.config.obter(
            "rede.user_agent_navegador",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        self.timeout = self.config.obter("rede.timeout_requisicao", 30)
        self.sessao = self._criar_sessao()
        self.sitemaps_visitados: Set[str] = set()

    def _criar_sessao(self) -> requests.Session:
        sessao = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        sessao.mount("http://", adapter)
        sessao.mount("https://", adapter)
        return sessao

    def _descomprimir_se_necessario(self, conteudo: bytes) -> bytes:
        if conteudo.startswith(b"\x1f\x8b"):
            try:
                return gzip.decompress(conteudo)
            except Exception:
                return conteudo
        return conteudo

    def _eh_sub_sitemap_relevante(self, sub_url: str) -> bool:
        match_ano = re.search(r"(?:/|-|_)(20\d{2})(?:/|-|_|\.|$)", sub_url)
        if match_ano:
            ano = int(match_ano.group(1))
            if ano < 2015 or ano > 2025:
                return False
        return True

    def coletar_urls_sitemap(
        self,
        sitemap_url: str,
        limite_total: Optional[int] = None,
        profundidade: int = 0,
        callback_lote: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        resultados: List[Dict[str, Any]] = []
        if profundidade > 3 or sitemap_url in self.sitemaps_visitados:
            return resultados

        self.sitemaps_visitados.add(sitemap_url)
        urls_vistas: Set[str] = set()

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/xml,application/rss+xml,application/gzip,*/*;q=0.8",
            "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        }

        try:
            resp = self.sessao.get(sitemap_url, headers=headers, timeout=self.timeout)
            if resp.status_code != 200:
                if resp.status_code == 429:
                    import time
                    time.sleep(10.0)
                self.logger.warning(
                    f"Sitemap {sitemap_url} retornou status HTTP {resp.status_code} ({resp.reason})"
                )
                return resultados

            conteudo = self._descomprimir_se_necessario(resp.content)

            if b"<rss" in conteudo or b"<feed" in conteudo or "rss" in sitemap_url:
                res_feed = self._processar_feed_rss(conteudo, limite_total)
                if callback_lote and res_feed:
                    callback_lote(res_feed)
                return res_feed

            if b"<?xml" in conteudo or b"<urlset" in conteudo or b"<sitemapindex" in conteudo or b"<sitemap" in conteudo:
                return self._processar_xml_sitemap(conteudo, sitemap_url, limite_total, profundidade, callback_lote)

            res_html = self._processar_html_secao(resp.text, sitemap_url, limite_total)
            if callback_lote and res_html:
                callback_lote(res_html)
            return res_html

        except Exception as e:
            self.logger.warning(f"Coleta de sitemap/secao {sitemap_url}: {e}")

        return resultados

    def coletar_sitemaps_historicos_cdx(
        self,
        sitemap_url: str,
        ano_inicio: int = 2015,
        ano_fim: int = 2025,
        limite_total: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        from src.coleta.cdx import ColetorCDX
        import random
        import time

        resultados: List[Dict[str, Any]] = []
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        params = {
            "url": sitemap_url,
            "output": "json",
            "fl": "timestamp,statuscode",
            "filter": "statuscode:200",
            "collapse": "timestamp:4",
            "from": f"{ano_inicio}0101",
            "to": f"{ano_fim}1231"
        }
        try:
            with ColetorCDX._lock:
                agora = time.time()
                if agora < ColetorCDX._cooldown_ate:
                    espera_cd = ColetorCDX._cooldown_ate - agora
                    time.sleep(espera_cd)
                    agora = time.time()

                decorrido = agora - ColetorCDX._ultimo_acesso
                intervalo = 3.2 + random.uniform(0.3, 0.8)
                if decorrido < intervalo:
                    time.sleep(intervalo - decorrido)
                ColetorCDX._ultimo_acesso = time.time()

                resp_cdx = self.sessao.get(
                    "https://web.archive.org/cdx/search/cdx",
                    params=params,
                    headers=headers,
                    timeout=self.timeout
                )
            if resp_cdx.status_code != 200:
                return resultados

            linhas = resp_cdx.json()
            if not linhas or len(linhas) <= 1:
                return resultados

            timestamps = [l[0] for l in linhas[1:] if len(l) > 0 and len(l[0]) >= 4]
            for ts in timestamps:
                if limite_total is not None and len(resultados) >= limite_total:
                    break
                url_wayback_sitemap = f"https://web.archive.org/web/{ts}id_/{sitemap_url}"
                with ColetorCDX._lock:
                    agora = time.time()
                    if agora < ColetorCDX._cooldown_ate:
                        time.sleep(ColetorCDX._cooldown_ate - agora)
                        agora = time.time()

                    decorrido = agora - ColetorCDX._ultimo_acesso
                    intervalo = 3.2 + random.uniform(0.3, 0.8)
                    if decorrido < intervalo:
                        time.sleep(intervalo - decorrido)
                    ColetorCDX._ultimo_acesso = time.time()
                    res_sm = self.coletar_urls_sitemap(url_wayback_sitemap, limite_total)

                for item in res_sm:
                    item["fonte_coleta"] = "sitemap_historico"
                    resultados.append(item)
        except Exception as e:
            self.logger.warning(f"Coleta de sitemaps historicos via CDX para {sitemap_url}: {e}")

        return resultados

    def _processar_xml_sitemap(
        self,
        conteudo: bytes,
        sitemap_url: str,
        limite_total: Optional[int],
        profundidade: int,
        callback_lote: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        resultados: List[Dict[str, Any]] = []
        urls_vistas: Set[str] = set()
        sub_sitemaps: List[str] = []
        urls_diretas: List[Tuple[str, Optional[str]]] = []

        try:
            root = ET.fromstring(conteudo)
            for elem in root.iter():
                tag_local = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
                if tag_local == "sitemap":
                    for filho in elem:
                        filho_tag = filho.tag.split("}")[-1] if "}" in filho.tag else filho.tag
                        if filho_tag == "loc" and filho.text:
                            sub_sitemaps.append(filho.text.strip())
                elif tag_local == "url":
                    loc_val = None
                    lastmod_val = None
                    for filho in elem:
                        filho_tag = filho.tag.split("}")[-1] if "}" in filho.tag else filho.tag
                        if filho_tag == "loc" and filho.text:
                            loc_val = filho.text.strip()
                        elif filho_tag in ("lastmod", "publication_date") and filho.text:
                            lastmod_val = filho.text.strip()[:10]
                    if loc_val:
                        urls_diretas.append((loc_val, lastmod_val))
        except Exception:
            texto_xml = conteudo.decode("utf-8", errors="ignore")
            matches_sitemap = re.findall(r"<sitemap>.*?<loc>\s*(https?://[^\s<]+)\s*</loc>.*?</sitemap>", texto_xml, re.DOTALL | re.IGNORECASE)
            sub_sitemaps.extend(matches_sitemap)

            matches_urls = re.findall(r"<url>(.*?)</url>", texto_xml, re.DOTALL | re.IGNORECASE)
            for bloco in matches_urls:
                m_loc = re.search(r"<loc>\s*(https?://[^\s<]+)\s*</loc>", bloco, re.IGNORECASE)
                m_date = re.search(r"<(?:lastmod|publication_date)>\s*([^\s<]+)\s*</", bloco, re.IGNORECASE)
                if m_loc:
                    loc_str = m_loc.group(1).strip()
                    dt_str = m_date.group(1).strip()[:10] if m_date else None
                    urls_diretas.append((loc_str, dt_str))

            if not sub_sitemaps and not urls_diretas:
                raw_locs = re.findall(r"<loc>\s*(https?://[^\s<]+)\s*</loc>", texto_xml, re.IGNORECASE)
                for l in raw_locs:
                    if ".xml" in l:
                        sub_sitemaps.append(l.strip())
                    else:
                        urls_diretas.append((l.strip(), None))

        itens_diretos_buffer: List[Dict[str, Any]] = []
        for url_artigo, dt in urls_diretas:
            if limite_total is not None and len(resultados) >= limite_total:
                break
            if not self.filtro.eh_url_valida(url_artigo, dt):
                continue
            if url_artigo in urls_vistas:
                continue

            urls_vistas.add(url_artigo)
            eixo = self.filtro.classificar_eixo_tematico(url_artigo)
            item_obj = {
                "url": url_artigo,
                "fonte_coleta": "sitemap",
                "timestamp_cdx": None,
                "data_prevista": dt,
                "eixo_tematico": eixo
            }
            resultados.append(item_obj)
            itens_diretos_buffer.append(item_obj)

        if callback_lote and itens_diretos_buffer:
            callback_lote(itens_diretos_buffer)

        if sub_sitemaps:
            sub_sitemaps_alvo = [s for s in sub_sitemaps if self._eh_sub_sitemap_relevante(s)]
            lim_sub = None if limite_total is None else max(20, (limite_total - len(resultados)) // max(1, len(sub_sitemaps_alvo)))
            for sub in sub_sitemaps_alvo:
                if limite_total is not None and len(resultados) >= limite_total:
                    break
                res_sub = self.coletar_urls_sitemap(sub, lim_sub, profundidade + 1, callback_lote)
                for item in res_sub:
                    if item["url"] not in urls_vistas:
                        urls_vistas.add(item["url"])
                        resultados.append(item)

        return resultados

    def _processar_html_secao(self, html: str, url_origem: str, limite_total: Optional[int] = None) -> List[Dict[str, Any]]:
        resultados = []
        urls_vistas = set()
        soup = BeautifulSoup(html, "html.parser")
        dominio_origem = urlparse(url_origem).netloc

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if not href.startswith("http"):
                href = urljoin(url_origem, href)

            parsed = urlparse(href)
            if dominio_origem not in parsed.netloc and not parsed.netloc.endswith(dominio_origem):
                continue

            if not self.filtro.eh_url_valida(href):
                continue

            if href in urls_vistas:
                continue

            urls_vistas.add(href)
            eixo = self.filtro.classificar_eixo_tematico(href, a.get_text())
            resultados.append({
                "url": href,
                "fonte_coleta": "secao_portal",
                "timestamp_cdx": None,
                "data_prevista": None,
                "eixo_tematico": eixo
            })

            if limite_total is not None and len(resultados) >= limite_total:
                break

        return resultados

    def _processar_feed_rss(self, conteudo: bytes, limite_total: Optional[int] = None) -> List[Dict[str, Any]]:
        resultados = []
        urls_vistas = set()
        try:
            soup = BeautifulSoup(conteudo, "xml")
        except Exception:
            soup = BeautifulSoup(conteudo, "html.parser")

        for item in soup.find_all(["item", "entry"]):
            link_tag = item.find("link")
            guid_tag = item.find("guid")

            url_texto = None
            if link_tag:
                url_texto = link_tag.get_text(strip=True) or link_tag.get("href")
            if not url_texto and guid_tag:
                guid_str = guid_tag.get_text(strip=True)
                if guid_str.startswith("http"):
                    url_texto = guid_str

            if not url_texto or not self.filtro.eh_url_valida(url_texto):
                continue

            if url_texto in urls_vistas:
                continue

            urls_vistas.add(url_texto)
            pub_date = item.find(["pubDate", "published", "updated", "dc:date"])
            dt = pub_date.get_text(strip=True)[:10] if pub_date else None
            eixo = self.filtro.classificar_eixo_tematico(url_texto)

            resultados.append({
                "url": url_texto,
                "fonte_coleta": "rss",
                "timestamp_cdx": None,
                "data_prevista": dt,
                "eixo_tematico": eixo
            })

            if limite_total is not None and len(resultados) >= limite_total:
                break

        return resultados
