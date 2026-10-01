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

    def _agrupar_por_janela_temporal(self, materias: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        if not materias:
            return []

        materias_ordenadas = sorted(materias, key=lambda m: m["data_publicacao"])
        janelas = []
        janela_atual = []
        data_referencia = None

        for mat in materias_ordenadas:
            try:
                dt_mat = datetime.strptime(mat["data_publicacao"][:10], "%Y-%m-%d")
            except Exception:
                dt_mat = datetime.now()

            if data_referencia is None:
                data_referencia = dt_mat
                janela_atual = [mat]
            else:
                if (dt_mat - data_referencia).days <= self.janela_dias:
                    janela_atual.append(mat)
                else:
                    if len(janela_atual) >= 2:
                        janelas.append(janela_atual)
                    data_referencia = dt_mat
                    janela_atual = [mat]

        if len(janela_atual) >= 2:
            janelas.append(janela_atual)

        return janelas

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
        todas_materias = self.banco.obter_todas_materias_com_embedding()
        if len(todas_materias) < 2:
            return {"total_clusters": 0, "materias_agrupadas": 0, "total_processadas": len(todas_materias)}

        janelas = self._agrupar_por_janela_temporal(todas_materias)
        proximo_evento_id = 1
        atualizacoes_materias: List[Tuple[int, int]] = []
        total_clusters_validos = 0
        total_materias_agrupadas = 0

        for janela in janelas:
            vetores = [GeradorEmbeddings.desserializar_vetor(m["vetor"]) for m in janela]
            X = np.array(vetores, dtype=np.float32)

            normas = np.linalg.norm(X, axis=1, keepdims=True)
            normas[normas == 0] = 1.0
            X_norm = X / normas

            clusters_finais = []

            if len(janela) >= 5:
                try:
                    clusterizador = HDBSCAN(
                        min_cluster_size=self.min_cluster_size,
                        min_samples=self.min_samples,
                        metric="euclidean",
                        cluster_selection_epsilon=self.cluster_selection_epsilon
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
            for mats_cluster in clusters_finais:
                ev_id = proximo_evento_id
                proximo_evento_id += 1

                for m in mats_cluster:
                    mats_agrupadas_ids.add(m["id"])
                    atualizacoes_materias.append((m["id"], ev_id))

                veiculos_ids = set(m["veiculo_id"] for m in mats_cluster)
                datas = [m["data_publicacao"] for m in mats_cluster]
                sentimentos = [m["sentimento_polaridade"] for m in mats_cluster]
                eixos = [m["eixo_tematico"] for m in mats_cluster if m.get("eixo_tematico")]
                eixo_pred = max(set(eixos), key=eixos.count) if eixos else "politico"

                divergencia = float(np.std(sentimentos)) if len(sentimentos) > 1 else 0.0
                titulo_rep = mats_cluster[0]["titulo"]

                self.banco.salvar_resumo_cluster(
                    evento_id=ev_id,
                    nome=titulo_rep,
                    dt_ini=min(datas),
                    dt_fim=max(datas),
                    tot_mat=len(mats_cluster),
                    tot_veic=len(veiculos_ids),
                    eixo=eixo_pred,
                    div_sent=round(divergencia, 4)
                )

                total_clusters_validos += 1
                total_materias_agrupadas += len(mats_cluster)

            for m in janela:
                if m["id"] not in mats_agrupadas_ids:
                    atualizacoes_materias.append((m["id"], -1))

            if callback_progresso:
                callback_progresso(len(janela))

        self.banco.atualizar_clusters_em_lote(atualizacoes_materias)
        self.logger.info(
            f"Clustering finalizado: {total_clusters_validos} clusters criados, "
            f"{total_materias_agrupadas} materias associadas."
        )

        return {
            "total_clusters": total_clusters_validos,
            "materias_agrupadas": total_materias_agrupadas,
            "total_processadas": len(todas_materias)
        }
