import gc
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.cluster import HDBSCAN
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.banco import BancoDados
from src.nlp.embeddings import GeradorEmbeddings


class AgrupadorEventos:
    def __init__(self, banco: BancoDados):
        self.banco = banco
        self.config = obter_configuracao()
        self.logger = obter_logger("clustering")
        self.janela_dias = self.config.obter("clustering.janela_dias", 5)
        self.min_cluster_size = self.config.obter("clustering.min_cluster_size", 2)
        self.min_samples = self.config.obter("clustering.min_samples", 1)
        self.cluster_selection_epsilon = self.config.obter("clustering.cluster_selection_epsilon", 0.35)
        self.similaridade_minima = self.config.obter("clustering.similaridade_minima", 0.65)

    def _gerar_intervalos_temporais(self) -> List[Tuple[str, str]]:
        datas = self.banco.obter_datas_distintas_materias()
        if not datas:
            return []

        intervalos = []
        data_ref = None
        dt_fim = None

        for d_str in datas:
            try:
                dt = datetime.strptime(d_str[:10], "%Y-%m-%d")
            except Exception:
                continue

            if data_ref is None:
                data_ref = dt
                dt_fim = data_ref + timedelta(days=self.janela_dias)
            elif dt > dt_fim:
                intervalos.append((data_ref.strftime("%Y-%m-%d"), dt_fim.strftime("%Y-%m-%d")))
                data_ref = dt
                dt_fim = data_ref + timedelta(days=self.janela_dias)

        if data_ref is not None and dt_fim is not None:
            intervalos.append((data_ref.strftime("%Y-%m-%d"), dt_fim.strftime("%Y-%m-%d")))

        return intervalos

    def _agrupar_por_similaridade_cosseno(self, janela: List[Dict[str, Any]], X_norm: np.ndarray) -> List[List[Dict[str, Any]]]:
        n = len(janela)
        visitados = [False] * n
        clusters = []

        matriz_sim = np.dot(X_norm, X_norm.T)

        for i in range(n):
            if visitados[i]:
                continue
            componente = [i]
            visitados[i] = True
            for j in range(i + 1, n):
                if not visitados[j] and matriz_sim[i, j] >= self.similaridade_minima:
                    visitados[j] = True
                    componente.append(j)

            if len(componente) >= self.min_cluster_size:
                clusters.append([janela[idx] for idx in componente])

        return clusters

    def executar(self, callback_progresso=None) -> Dict[str, Any]:
        self.banco.limpar_clusters_eventos()
        intervalos = self._gerar_intervalos_temporais()
        if not intervalos:
            return {"total_clusters": 0, "materias_agrupadas": 0, "total_processadas": 0}

        proximo_evento_id = 1
        total_clusters_validos = 0
        total_materias_agrupadas = 0
        total_processadas = 0

        for dt_ini, dt_fim in intervalos:
            janela = self.banco.obter_materias_para_clusterizacao_por_janela(dt_ini, dt_fim)
            qtd_janela = len(janela)
            if qtd_janela == 0:
                continue

            total_processadas += qtd_janela

            if qtd_janela < self.min_cluster_size:
                atualizacoes_isoladas = [(m["id"], -1) for m in janela]
                self.banco.atualizar_clusters_em_lote(atualizacoes_isoladas)
                if callback_progresso:
                    callback_progresso(qtd_janela)
                continue

            vetores = [GeradorEmbeddings.desserializar_vetor(m["vetor"]) for m in janela]
            X = np.array(vetores, dtype=np.float32)

            normas = np.linalg.norm(X, axis=1, keepdims=True)
            normas[normas == 0] = 1.0
            X_norm = X / normas

            clusters_finais = []

            if qtd_janela >= 5:
                try:
                    clusterizador = HDBSCAN(
                        min_cluster_size=self.min_cluster_size,
                        min_samples=self.min_samples,
                        metric="euclidean",
                        cluster_selection_epsilon=self.cluster_selection_epsilon,
                        copy=True
                    )
                    labels = clusterizador.fit_predict(X_norm)
                    mapa_lbl = {}
                    for idx_m, lbl in enumerate(labels):
                        if lbl >= 0:
                            if lbl not in mapa_lbl:
                                mapa_lbl[lbl] = []
                            mapa_lbl[lbl].append(janela[idx_m])
                    clusters_finais = list(mapa_lbl.values())
                except Exception:
                    clusters_finais = self._agrupar_por_similaridade_cosseno(janela, X_norm)
            else:
                clusters_finais = self._agrupar_por_similaridade_cosseno(janela, X_norm)

            mats_agrupadas_ids = set()
            resumos_clusters_janela = []
            atualizacoes_janela = []

            for mats_cluster in clusters_finais:
                ev_id = proximo_evento_id
                proximo_evento_id += 1

                for m in mats_cluster:
                    mats_agrupadas_ids.add(m["id"])
                    atualizacoes_janela.append((m["id"], ev_id))

                veiculos_ids = set(m["veiculo_id"] for m in mats_cluster)
                datas = [m["data_publicacao"] for m in mats_cluster]
                sentimentos = [float(m.get("sentimento_polaridade", 0.0) or 0.0) for m in mats_cluster]
                eixos = [m["eixo_tematico"] for m in mats_cluster if m.get("eixo_tematico")]
                eixo_pred = max(set(eixos), key=eixos.count) if eixos else "politico"

                divergencia = float(np.std(sentimentos)) if len(sentimentos) > 1 else 0.0
                titulo_rep = mats_cluster[0]["titulo"]

                resumos_clusters_janela.append((
                    ev_id,
                    titulo_rep,
                    min(datas),
                    max(datas),
                    len(mats_cluster),
                    len(veiculos_ids),
                    eixo_pred,
                    round(divergencia, 4)
                ))

                total_clusters_validos += 1
                total_materias_agrupadas += len(mats_cluster)

            for m in janela:
                if m["id"] not in mats_agrupadas_ids:
                    atualizacoes_janela.append((m["id"], -1))

            self.banco.salvar_resumos_clusters_em_lote(resumos_clusters_janela)
            self.banco.atualizar_clusters_em_lote(atualizacoes_janela)

            if callback_progresso:
                callback_progresso(qtd_janela)

            del janela
            del vetores
            del X
            del X_norm
            del clusters_finais
            del atualizacoes_janela
            del resumos_clusters_janela
            gc.collect()

        self.logger.info(
            f"Clustering finalizado: {total_clusters_validos} clusters criados, "
            f"{total_materias_agrupadas} materias associadas."
        )

        return {
            "total_clusters": total_clusters_validos,
            "materias_agrupadas": total_materias_agrupadas,
            "total_processadas": total_processadas
        }
