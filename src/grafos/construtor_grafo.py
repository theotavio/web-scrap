from pathlib import Path
from typing import Any, Dict, List, Tuple
import networkx as nx
import numpy as np
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.banco import BancoDados


class ConstrutorGrafos:
    def __init__(self, banco: BancoDados):
        self.banco = banco
        self.config = obter_configuracao()
        self.logger = obter_logger("grafos")
        self.dir_relatorios = Path(self.config.obter("geral.diretorio_relatorios", "relatorios")) / "grafos"
        self.dir_relatorios.mkdir(parents=True, exist_ok=True)

    def construir_grafo_bipartido(self) -> nx.Graph:
        con = self.banco.obter_conexao()
        cur = con.cursor()

        cur.execute("""
            SELECT m.id, m.veiculo_id, m.evento_id, m.sentimento_polaridade,
                   m.score_clickbait, v.nome as veiculo_nome, v.tipo as veiculo_tipo,
                   ce.nome_evento, ce.eixo_predominante, ce.divergencia_sentimento
            FROM materias m
            JOIN veiculos v ON m.veiculo_id = v.id
            JOIN clusters_eventos ce ON m.evento_id = ce.id
            WHERE m.evento_id > -1
        """)
        linhas = [dict(row) for row in cur.fetchall()]
        con.close()

        G = nx.Graph()

        veiculos_adicionados = set()
        eventos_adicionados = set()
        arestas_acumuladas: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

        for l in linhas:
            no_veiculo = f"veiculo_{l['veiculo_nome']}"
            no_evento = f"evento_{l['evento_id']}"

            if no_veiculo not in veiculos_adicionados:
                G.add_node(
                    no_veiculo,
                    rotulo=l["veiculo_nome"],
                    tipo_no="veiculo",
                    tipo_editorial=l["veiculo_tipo"],
                    bipartite=0
                )
                veiculos_adicionados.add(no_veiculo)

            if no_evento not in eventos_adicionados:
                G.add_node(
                    no_evento,
                    rotulo=l["nome_evento"][:50],
                    tipo_no="evento",
                    eixo=l["eixo_predominante"],
                    divergencia=float(l["divergencia_sentimento"]),
                    bipartite=1
                )
                eventos_adicionados.add(no_evento)

            chave_aresta = (no_veiculo, no_evento)
            if chave_aresta not in arestas_acumuladas:
                arestas_acumuladas[chave_aresta] = []
            arestas_acumuladas[chave_aresta].append(l)

        for (no_v, no_e), itens in arestas_acumuladas.items():
            qtd = len(itens)
            sent_med = float(np.mean([it["sentimento_polaridade"] for it in itens]))
            cb_med = float(np.mean([it["score_clickbait"] for it in itens]))

            G.add_edge(
                no_v,
                no_e,
                weight=qtd,
                sentimento_medio=round(sent_med, 4),
                clickbait_medio=round(cb_med, 4)
            )

        return G

    def construir_projecao_veiculos(self, G_bip: nx.Graph) -> nx.Graph:
        G_proj = nx.Graph()

        nos_veiculos = [n for n, d in G_bip.nodes(data=True) if d.get("bipartite") == 0]
        for n in nos_veiculos:
            d = G_bip.nodes[n]
            G_proj.add_node(
                n,
                rotulo=d.get("rotulo"),
                tipo_editorial=d.get("tipo_editorial")
            )

        for i in range(len(nos_veiculos)):
            for j in range(i + 1, len(nos_veiculos)):
                v1 = nos_veiculos[i]
                v2 = nos_veiculos[j]

                vizinhos_v1 = set(G_bip.neighbors(v1))
                vizinhos_v2 = set(G_bip.neighbors(v2))
                eventos_comuns = list(vizinhos_v1.intersection(vizinhos_v2))

                if not eventos_comuns:
                    continue

                diferencas_sentimento = []
                for ev in eventos_comuns:
                    s1 = G_bip[v1][ev].get("sentimento_medio", 0.0)
                    s2 = G_bip[v2][ev].get("sentimento_medio", 0.0)
                    diferencas_sentimento.append(abs(s1 - s2))

                dist_media = float(np.mean(diferencas_sentimento)) if diferencas_sentimento else 0.0
                similaridade_enquadramento = max(0.0, 1.0 - (dist_media / 2.0))
                peso_final = len(eventos_comuns) * similaridade_enquadramento

                G_proj.add_edge(
                    v1,
                    v2,
                    weight=round(peso_final, 4),
                    eventos_comuns=len(eventos_comuns),
                    distancia_sentimento=round(dist_media, 4),
                    similaridade_enquadramento=round(similaridade_enquadramento, 4)
                )

        return G_proj

    def exportar_gexf(self, G: nx.Graph, nome_arquivo: str) -> str:
        caminho = self.dir_relatorios / nome_arquivo
        nx.write_gexf(G, str(caminho))
        self.logger.info(f"Grafo exportado para Gephi em: {caminho}")
        return str(caminho)

    def executar(self) -> Dict[str, Any]:
        G_bip = self.construir_grafo_bipartido()
        G_proj = self.construir_projecao_veiculos(G_bip)

        caminho_bip = self.exportar_gexf(G_bip, "rede_bipartida.gexf")
        caminho_proj = self.exportar_gexf(G_proj, "rede_veiculos.gexf")

        metricas = {
            "bipartido_nos": G_bip.number_of_nodes(),
            "bipartido_arestas": G_bip.number_of_edges(),
            "projecao_nos": G_proj.number_of_nodes(),
            "projecao_arestas": G_proj.number_of_edges(),
            "caminho_gexf_bipartido": caminho_bip,
            "caminho_gexf_projecao": caminho_proj
        }

        self.logger.info(f"Processamento de grafos concluido: {metricas}")
        return metricas
