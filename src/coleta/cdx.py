import random
import threading
import time
from typing import Any, Dict, List, Optional, Set
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.coleta.filtros import FiltroUrls


class ColetorCDX:
    _lock = threading.Lock()
    _ultimo_acesso = 0.0
    _cooldown_ate = 0.0

    def __init__(self):
        self.config = obter_configuracao()
        self.logger = obter_logger("coleta")
        self.filtro = FiltroUrls()
        self.url_base = self.config.obter("cdx.url_base", "https://web.archive.org/cdx/search/cdx")
        self.url_base_https = self.config.obter("cdx.url_base_https", "https://web.archive.org/cdx/search/cdx")
        self.user_agent = self.config.obter(
            "rede.user_agent_academico",
            "ProjetoPesquisaJornalismoUFC/1.0 (pesquisa.academica; contato: projeto@pesquisa.br)"
        )
        self.timeout = self.config.obter("rede.timeout_requisicao", 45)
        self.limite_requisicao = self.config.obter("cdx.limite_por_requisicao", 1000)
        self.pausa = self.config.obter("cdx.pausa_segundos", 0.2)
        self.sessao = self._criar_sessao()

    def _criar_sessao(self) -> requests.Session:
        sessao = requests.Session()
        retries = Retry(
            total=0,
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
        sessao.mount("http://", adapter)
        sessao.mount("https://", adapter)
        return sessao

    def _requisitar_cdx_seguro(
        self,
        endpoint: str,
        parametros: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Optional[requests.Response]:
        endpoint_tentativa = endpoint
        backoff_lista = [8, 15, 30, 60, 90, 120]

        for tentativa in range(6):
            with ColetorCDX._lock:
                agora = time.time()
                if agora < ColetorCDX._cooldown_ate:
                    espera_cd = ColetorCDX._cooldown_ate - agora
                    time.sleep(espera_cd)
                    agora = time.time()

                decorrido = agora - ColetorCDX._ultimo_acesso
                intervalo_base = 3.2 + random.uniform(0.3, 0.8)
                if decorrido < intervalo_base:
                    time.sleep(intervalo_base - decorrido)
                ColetorCDX._ultimo_acesso = time.time()

                try:
                    resp = self.sessao.get(
                        endpoint_tentativa,
                        params=parametros,
                        headers=headers,
                        timeout=self.timeout
                    )
                except Exception as e:
                    resp = None
                    erro_conexao = e
                else:
                    erro_conexao = None

            if resp is not None:
                if resp.status_code == 200:
                    return resp

                if resp.status_code in (400, 403, 404):
                    self.logger.error(
                        f"CDX requisicao abortada com status HTTP {resp.status_code} para {parametros.get('url')}"
                    )
                    return None

                tempo_espera = backoff_lista[min(tentativa, len(backoff_lista) - 1)] + random.uniform(1.0, 3.0)
                with ColetorCDX._lock:
                    ColetorCDX._cooldown_ate = max(ColetorCDX._cooldown_ate, time.time() + tempo_espera)

                self.logger.warning(
                    f"CDX {endpoint_tentativa} retornou HTTP {resp.status_code} na tentativa {tentativa + 1}/6. "
                    f"Ativando cooldown global de {tempo_espera:.1f}s..."
                )
                time.sleep(tempo_espera)
                endpoint_tentativa = "http://web.archive.org/cdx/search/cdx" if "https:" in endpoint_tentativa else "https://web.archive.org/cdx/search/cdx"
            else:
                tempo_espera = backoff_lista[min(tentativa, len(backoff_lista) - 1)] + random.uniform(1.0, 3.0)
                with ColetorCDX._lock:
                    ColetorCDX._cooldown_ate = max(ColetorCDX._cooldown_ate, time.time() + tempo_espera)

                self.logger.warning(
                    f"CDX conexao falhou {endpoint_tentativa} na tentativa {tentativa + 1}/6: {erro_conexao}. "
                    f"Ativando cooldown global de {tempo_espera:.1f}s..."
                )
                time.sleep(tempo_espera)
                endpoint_tentativa = "http://web.archive.org/cdx/search/cdx" if "https:" in endpoint_tentativa else "https://web.archive.org/cdx/search/cdx"

        self.logger.error(
            f"CDX falha definitiva apos 6 tentativas para url={parametros.get('url')} "
            f"periodo={parametros.get('from')}-{parametros.get('to')}"
        )
        return None

    def _coletar_periodo(
        self,
        prefixo: str,
        data_de: str,
        data_ate: str,
        limite_total: Optional[int],
        urls_vistas: Set[str],
        callback_lote: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        resultados_periodo: List[Dict[str, Any]] = []
        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }

        endpoint_atual = self.url_base
        resume_key: Optional[str] = None
        max_paginas = 200 if limite_total is None else max(1, limite_total // 1000)

        for pagina in range(max_paginas):
            if limite_total is not None and len(urls_vistas) >= limite_total:
                break

            limite_req = self.limite_requisicao
            if limite_total is not None:
                restante = limite_total - len(urls_vistas)
                limite_req = min(self.limite_requisicao, restante)

            parametros = {
                "url": prefixo,
                "matchType": "prefix",
                "from": data_de,
                "to": data_ate,
                "output": "json",
                "fl": "original,timestamp,mimetype,statuscode",
                "filter": ["statuscode:200", "mimetype:text/html"],
                "collapse": "urlkey",
                "limit": limite_req,
                "showResumeKey": "true"
            }
            if resume_key:
                parametros["resumeKey"] = resume_key

            resp = self._requisitar_cdx_seguro(endpoint_atual, parametros, headers)
            if resp is None:
                if resume_key:
                    self.logger.error(
                        f"CDX paginacao interrompida prematuramente no periodo {data_de}-{data_ate} "
                        f"para prefixo {prefixo} na pagina {pagina + 1} apos coletar {len(resultados_periodo)} URLs."
                    )
                break

            texto_resp = resp.text.strip() if resp.text else ""
            if not texto_resp:
                break

            try:
                linhas = resp.json()
            except Exception as e:
                self.logger.warning(f"CDX retorno nao JSON para {prefixo} ({data_de}-{data_ate}): {e}")
                break

            if not linhas or len(linhas) <= 1:
                break

            proximo_resume_key = None
            linhas_dados = []

            if len(linhas[-1]) == 1 and isinstance(linhas[-1][0], str) and len(linhas[-1][0]) > 20:
                proximo_resume_key = linhas[-1][0]
                linhas_dados = linhas[1:-2] if len(linhas) > 2 and len(linhas[-2]) == 0 else linhas[1:-1]
            else:
                linhas_dados = linhas[1:]

            itens_pagina: List[Dict[str, Any]] = []
            for linha in linhas_dados:
                if len(linha) < 2:
                    continue

                url_original = linha[0]
                timestamp = linha[1]
                mime = linha[2] if len(linha) > 2 else "text/html"

                if mime not in ("text/html", "text/plain", "warc/revisit"):
                    continue

                data_prevista = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]}"
                if not self.filtro.eh_url_valida(url_original, data_prevista):
                    continue

                if url_original in urls_vistas:
                    continue

                urls_vistas.add(url_original)
                eixo = self.filtro.classificar_eixo_tematico(url_original)

                item_obj = {
                    "url": url_original,
                    "fonte_coleta": "cdx",
                    "timestamp_cdx": timestamp,
                    "data_prevista": data_prevista,
                    "eixo_tematico": eixo
                }
                resultados_periodo.append(item_obj)
                itens_pagina.append(item_obj)

                if limite_total is not None and len(urls_vistas) >= limite_total:
                    break

            if callback_lote and itens_pagina:
                callback_lote(itens_pagina)

            if not proximo_resume_key or proximo_resume_key == resume_key:
                break

            resume_key = proximo_resume_key
            if self.pausa > 0:
                time.sleep(self.pausa)

        return resultados_periodo

    def coletar_urls_prefixo(
        self,
        prefixo: str,
        ano_inicio: int = 2015,
        ano_fim: int = 2025,
        lista_anos: Optional[List[int]] = None,
        limite_total: Optional[int] = None,
        callback_lote: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        resultados: List[Dict[str, Any]] = []
        urls_vistas: Set[str] = set()

        anos = lista_anos if lista_anos else list(range(ano_inicio, ano_fim + 1))
        for ano in anos:
            if limite_total is not None and len(resultados) >= limite_total:
                break

            data_de = f"{ano}0101"
            data_ate = f"{ano}1231"

            res_ano = self._coletar_periodo(
                prefixo=prefixo,
                data_de=data_de,
                data_ate=data_ate,
                limite_total=limite_total,
                urls_vistas=urls_vistas,
                callback_lote=callback_lote
            )
            resultados.extend(res_ano)

        return resultados
