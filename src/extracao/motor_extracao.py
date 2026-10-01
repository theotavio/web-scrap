import concurrent.futures
from typing import Any, Dict, Optional, Tuple
from tqdm import tqdm
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.banco import BancoDados
from src.robos import VerificadorRobots
from src.anti_paywall import BypassCascataPaywall
from src.extracao.extrator import ExtratorArtigo


class MotorExtracao:
    def __init__(self, banco: BancoDados):
        self.banco = banco
        self.config = obter_configuracao()
        self.logger = obter_logger("extracao")
        self.robots = VerificadorRobots()
        self.paywall = BypassCascataPaywall()
        self.extrator = ExtratorArtigo()
        self.max_threads = self.config.obter("rede.max_threads_extracao", 8)

    def _processar_url(self, item_url: Dict[str, Any]) -> Tuple[int, str, Optional[int]]:
        url_id = item_url["id"]
        url = item_url["url"]
        veiculo_id = item_url["veiculo_id"]
        ts_cdx = item_url.get("timestamp_cdx")

        if not self.robots.pode_coletar(url):
            self.banco.atualizar_status_url(url_id, "bloqueado_robots")
            return url_id, "bloqueado_robots", None

        html, _, metadados = self.paywall.baixar_conteudo(url, timestamp_cdx=ts_cdx)
        if not html:
            self.banco.atualizar_status_url(url_id, "erro_download")
            return url_id, "erro_download", None

        dados_ld = metadados.get("dados_json_ld")
        artigo = self.extrator.extrair(html, url, dados_json_ld=dados_ld)
        if not artigo:
            self.banco.atualizar_status_url(url_id, "descartado_incompleto")
            return url_id, "descartado_incompleto", None

        artigo["url_id"] = url_id
        artigo["veiculo_id"] = veiculo_id

        materia_id = self.banco.inserir_materia(artigo)
        if materia_id is None:
            return url_id, "duplicado", None

        return url_id, "sucesso", materia_id

    def executar(self, limite: Optional[int] = None, callback_progresso=None) -> Dict[str, int]:
        urls_pendentes = self.banco.obter_urls_pendentes(limite=limite)
        total = len(urls_pendentes)
        if total == 0:
            return {"total": 0, "sucesso": 0, "falha": 0, "descartados": 0}

        estatisticas = {"total": total, "sucesso": 0, "falha": 0, "descartados": 0, "duplicados": 0}

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futuros = {executor.submit(self._processar_url, item): item for item in urls_pendentes}

            for futuro in concurrent.futures.as_completed(futuros):
                try:
                    _, status, _ = futuro.result()
                    if status == "sucesso":
                        estatisticas["sucesso"] += 1
                    elif status in ("descartado_incompleto", "bloqueado_robots"):
                        estatisticas["descartados"] += 1
                    elif status == "duplicado":
                        estatisticas["duplicados"] += 1
                    else:
                        estatisticas["falha"] += 1
                except Exception as e:
                    self.logger.error(f"Excecao no processamento de URL em thread: {e}")
                    estatisticas["falha"] += 1

                if callback_progresso:
                    callback_progresso(1)

        self.logger.info(f"Extracao finalizada: {estatisticas}")
        return estatisticas
