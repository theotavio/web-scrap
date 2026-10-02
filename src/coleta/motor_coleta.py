import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.banco import BancoDados
from src.coleta.cdx import ColetorCDX
from src.coleta.sitemaps import ColetorSitemaps


class MotorColeta:
    def __init__(self, banco: BancoDados):
        self.banco = banco
        self.config = obter_configuracao()
        self.logger = obter_logger("coleta")
        self.coletor_cdx = ColetorCDX()
        self.coletor_sitemaps = ColetorSitemaps()
        self.max_threads = self.config.obter("rede.max_threads_coleta", 6)
        self.ano_inicio = int(self.config.obter("geral.data_inicio", "2015-01-01").split("-")[0])
        self.ano_fim = int(self.config.obter("geral.data_fim", "2025-12-31").split("-")[0])

    def _obter_mapa_veiculos(self) -> Dict[str, int]:
        con = self.banco.obter_conexao()
        cur = con.cursor()
        cur.execute("SELECT codigo, id FROM veiculos")
        mapa = {row[0]: row[1] for row in cur.fetchall()}
        con.close()
        return mapa

    def _coletar_veiculo(
        self,
        cod_veiculo: str,
        veiculo_id: int,
        info_veiculo: Dict[str, Any],
        limite_por_veiculo: Optional[int] = None,
        fonte: str = "todas",
        ano_inicio: Optional[int] = None,
        ano_fim: Optional[int] = None,
        lista_anos: Optional[List[int]] = None
    ) -> Tuple[str, int, int]:
        total_cdx = 0
        total_sitemap = 0
        total_inseridos = 0
        lote_buffer: List[Tuple[int, str, str, Optional[str], Optional[str], Optional[str]]] = []
        urls_vistas = set()

        def salvar_buffer():
            nonlocal total_inseridos, lote_buffer
            if lote_buffer:
                ins = self.banco.inserir_urls_em_lote(lote_buffer)
                total_inseridos += ins
                lote_buffer = []

        def callback_sitemap_lote(lote_items):
            nonlocal total_inseridos, lote_buffer, urls_vistas
            for item in lote_items:
                u = item["url"]
                if u not in urls_vistas:
                    urls_vistas.add(u)
                    lote_buffer.append((
                        veiculo_id,
                        u,
                        item["fonte_coleta"],
                        item.get("timestamp_cdx"),
                        item.get("data_prevista"),
                        item.get("eixo_tematico")
                    ))
                    if len(lote_buffer) >= 1000:
                        salvar_buffer()

        sitemaps = info_veiculo.get("sitemaps", [])
        lim_por_sm = None
        if limite_por_veiculo is not None:
            lim_por_sm = max(50, limite_por_veiculo // max(1, len(sitemaps)))

        if fonte in ("sitemaps", "todas"):
            for sm_url in sitemaps:
                urls_sm = self.coletor_sitemaps.coletar_urls_sitemap(
                    sm_url,
                    limite_total=lim_por_sm,
                    callback_lote=callback_sitemap_lote
                )
                total_sitemap += len(urls_sm)

            salvar_buffer()

        if fonte in ("cdx", "todas"):
            prefixos = info_veiculo.get("prefixos_cdx", [])
            ano_ini_padrao = int(info_veiculo.get("ano_inicio", self.ano_inicio))
            ano_ini_veic = ano_inicio if ano_inicio is not None else ano_ini_padrao
            ano_fim_veic = ano_fim if ano_fim is not None else self.ano_fim
            lim_por_prefixo = None
            if limite_por_veiculo is not None:
                lim_por_prefixo = max(50, limite_por_veiculo // max(1, len(prefixos)))

            for prefixo in prefixos:
                if limite_por_veiculo is not None and total_cdx >= limite_por_veiculo:
                    break

                urls_cdx = self.coletor_cdx.coletar_urls_prefixo(
                    prefixo=prefixo,
                    ano_inicio=ano_ini_veic,
                    ano_fim=ano_fim_veic,
                    lista_anos=lista_anos,
                    limite_total=lim_por_prefixo,
                    callback_lote=callback_sitemap_lote
                )
                for item in urls_cdx:
                    u = item["url"]
                    if u not in urls_vistas:
                        urls_vistas.add(u)
                        lote_buffer.append((
                            veiculo_id,
                            u,
                            item["fonte_coleta"],
                            item.get("timestamp_cdx"),
                            item.get("data_prevista"),
                            item.get("eixo_tematico")
                        ))
                        if len(lote_buffer) >= 1000:
                            salvar_buffer()

                total_cdx += len(urls_cdx)

            salvar_buffer()

            for sm_url in sitemaps:
                urls_sm_hist = self.coletor_sitemaps.coletar_sitemaps_historicos_cdx(
                    sitemap_url=sm_url,
                    ano_inicio=ano_ini_veic,
                    ano_fim=ano_fim_veic,
                    limite_total=lim_por_sm
                )
                for item in urls_sm_hist:
                    u = item["url"]
                    if u not in urls_vistas:
                        urls_vistas.add(u)
                        lote_buffer.append((
                            veiculo_id,
                            u,
                            item["fonte_coleta"],
                            item.get("timestamp_cdx"),
                            item.get("data_prevista"),
                            item.get("eixo_tematico")
                        ))
                        if len(lote_buffer) >= 1000:
                            salvar_buffer()

                total_sitemap += len(urls_sm_hist)

            salvar_buffer()

        total_encontradas = total_cdx + total_sitemap
        self.logger.info(
            f"Veículo {cod_veiculo} coletado: {total_encontradas} URLs encontradas "
            f"({total_cdx} CDX, {total_sitemap} Sitemaps/Históricos), {total_inseridos} novas inseridas no SQLite."
        )
        return cod_veiculo, total_encontradas, total_inseridos

    def executar(
        self,
        limite_por_veiculo: Optional[int] = None,
        fonte: str = "todas",
        veiculo: Optional[str] = None,
        ano_inicio: Optional[int] = None,
        ano_fim: Optional[int] = None,
        lista_anos: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        mapa_veiculos = self._obter_mapa_veiculos()
        veiculos_cfg = self.config.obter("veiculos", {})

        if veiculo:
            termos = [t.strip().lower() for t in veiculo.split(",") if t.strip()]
            veiculos_filtrados = {}
            for t in termos:
                if t in ("tradicional", "tradicionais"):
                    for k, v in veiculos_cfg.items():
                        if v.get("tipo") == "tradicional":
                            veiculos_filtrados[k] = v
                elif t in ("digital", "digitais", "digital_nativo", "digitais_nativos"):
                    for k, v in veiculos_cfg.items():
                        if v.get("tipo") == "digital_nativo":
                            veiculos_filtrados[k] = v
                else:
                    for k, v in veiculos_cfg.items():
                        if t == k.lower() or t in k.lower() or t in v.get("nome", "").lower():
                            veiculos_filtrados[k] = v
            if veiculos_filtrados:
                veiculos_cfg = veiculos_filtrados

        resumo = {}
        tarefas = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            for cod, info in veiculos_cfg.items():
                v_id = mapa_veiculos.get(cod)
                if v_id is None:
                    continue
                tarefa = executor.submit(
                    self._coletar_veiculo,
                    cod,
                    v_id,
                    info,
                    limite_por_veiculo,
                    fonte,
                    ano_inicio,
                    ano_fim,
                    lista_anos
                )
                tarefas.append(tarefa)

            for tarefa in concurrent.futures.as_completed(tarefas):
                try:
                    cod, encontradas, inseridas = tarefa.result()
                    resumo[cod] = {"encontradas": encontradas, "inseridas": inseridas}
                except Exception as e:
                    self.logger.error(f"Erro na thread de coleta de veículo: {e}")

        return resumo
