import re
from datetime import datetime
from typing import Any, Dict, Optional, Tuple
from bs4 import BeautifulSoup
import trafilatura
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.coleta.filtros import FiltroUrls
from src.extracao.limpador import LimpadorTexto


class ExtratorArtigo:
    def __init__(self):
        self.config = obter_configuracao()
        self.logger = obter_logger("extracao")
        self.filtro = FiltroUrls()
        self.min_tamanho = self.config.obter("filtros.min_tamanho_corpo", 150)
        self.anos_eleitorais = set(self.config.obter("geral.anos_eleitorais", [2016, 2018, 2020, 2022, 2024]))

    def _extrair_data_html_ou_url(self, html: str, url: str, data_meta: Optional[str]) -> Tuple[str, int, int, int]:
        data_str = None

        if data_meta:
            match_iso = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(data_meta))
            if match_iso:
                data_str = f"{match_iso.group(1)}-{match_iso.group(2)}-{match_iso.group(3)}"

        if not data_str:
            match_url = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", url)
            if match_url:
                data_str = f"{match_url.group(1)}-{match_url.group(2)}-{match_url.group(3)}"

        if not data_str:
            match_url_alt = re.search(r"/(\d{4})/(\d{2})/", url)
            if match_url_alt:
                data_str = f"{match_url_alt.group(1)}-{match_url_alt.group(2)}-01"

        if not data_str:
            soup = BeautifulSoup(html[:5000], "html.parser")
            meta_date = (
                soup.find("meta", property="article:published_time")
                or soup.find("meta", {"name": "publication_date"})
                or soup.find("meta", {"name": "date"})
            )
            if meta_date and meta_date.get("content"):
                match_meta = re.search(r"(\d{4})-(\d{2})-(\d{2})", meta_date["content"])
                if match_meta:
                    data_str = f"{match_meta.group(1)}-{match_meta.group(2)}-{match_meta.group(3)}"

        if not data_str:
            data_str = datetime.now().strftime("%Y-%m-%d")

        ano = int(data_str[:4])
        mes = int(data_str[5:7])
        ano_eleitoral = 1 if ano in self.anos_eleitorais else 0

        return data_str, ano, mes, ano_eleitoral

    def extrair(self, html: str, url: str, dados_json_ld: Optional[Tuple] = None) -> Optional[Dict[str, Any]]:
        if not html:
            return None

        titulo = None
        corpo = None
        autor = None
        data_pub = None

        if dados_json_ld:
            titulo_ld, corpo_ld, data_ld, autor_ld = dados_json_ld
            if len(corpo_ld) >= self.min_tamanho:
                titulo = titulo_ld
                corpo = corpo_ld
                autor = autor_ld
                data_pub = data_ld

        if not corpo or len(corpo) < self.min_tamanho:
            trafilatura_texto = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                no_fallback=False
            )
            trafilatura_meta = trafilatura.extract_metadata(html)

            if trafilatura_texto and len(trafilatura_texto) >= self.min_tamanho:
                corpo = trafilatura_texto
                if trafilatura_meta:
                    titulo = titulo or trafilatura_meta.title
                    autor = autor or trafilatura_meta.author
                    data_pub = data_pub or trafilatura_meta.date

        if not corpo or len(corpo) < self.min_tamanho:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            paragrafos = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]
            if paragrafos:
                texto_soup = "\n\n".join(paragrafos)
                if len(texto_soup) >= self.min_tamanho:
                    corpo = texto_soup
                    if not titulo:
                        h1 = soup.find("h1")
                        titulo = h1.get_text(strip=True) if h1 else (soup.title.string if soup.title else "Sem Titulo")

        if not corpo or len(corpo) < self.min_tamanho:
            return None

        if not titulo:
            soup = BeautifulSoup(html[:3000], "html.parser")
            h1 = soup.find("h1")
            titulo = h1.get_text(strip=True) if h1 else (soup.title.string if soup.title else "Sem Titulo")

        titulo_limpo = LimpadorTexto.normalizar_unicode(titulo).strip()
        corpo_limpo = LimpadorTexto.remover_boilerplate(corpo)

        if len(corpo_limpo) < self.min_tamanho:
            return None

        resumo = LimpadorTexto.extrair_resumo(corpo_limpo)
        hash_conteudo = LimpadorTexto.gerar_hash_conteudo(titulo_limpo, corpo_limpo)
        data_final, ano, mes, ano_eleitoral = self._extrair_data_html_ou_url(html, url, data_pub)
        ano_inicio = int(self.config.obter("geral.data_inicio", "2015-01-01").split("-")[0])
        ano_fim = int(self.config.obter("geral.data_fim", "2025-12-31").split("-")[0])
        if ano < ano_inicio or ano > ano_fim:
            return None

        eixo_tematico = self.filtro.classificar_eixo_tematico(url, titulo_limpo + " " + corpo_limpo[:300])

        return {
            "url": url,
            "titulo": titulo_limpo,
            "corpo": corpo_limpo,
            "resumo": resumo,
            "data_publicacao": data_final,
            "ano": ano,
            "mes": mes,
            "ano_eleitoral": ano_eleitoral,
            "autor": autor,
            "eixo_tematico": eixo_tematico,
            "hash_conteudo": hash_conteudo,
            "tamanho_texto": len(corpo_limpo)
        }
