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

    def construir_subgrafo_macro_eventos(self, limite_eventos: int = 1000, min_veiculos: int = 5) -> Tuple[nx.Graph, Dict[str, Any]]:
        con = self.banco.obter_conexao()
        cur = con.cursor()

        cur.execute(f"""
            SELECT id, nome_evento, total_materias, total_veiculos, eixo_predominante, divergencia_sentimento
            FROM clusters_eventos
            WHERE total_veiculos >= {min_veiculos}
            ORDER BY total_veiculos DESC, total_materias DESC
            LIMIT {limite_eventos}
        """)
        eventos = cur.fetchall()
        eventos_ids = set(e[0] for e in eventos)

        cur.execute(f"""
            SELECT m.evento_id, m.veiculo_id, v.nome as veiculo_nome, v.tipo as veiculo_tipo,
                   AVG(m.sentimento_polaridade) as sent_med,
                   AVG(m.score_clickbait) as cb_med,
                   COUNT(m.id) as qtd_materias
            FROM materias m
            JOIN veiculos v ON m.veiculo_id = v.id
            WHERE m.evento_id IN ({','.join(str(i) for i in eventos_ids)})
            GROUP BY m.evento_id, m.veiculo_id
        """)
        arestas = cur.fetchall()
        con.close()

        G = nx.Graph()
        for e in eventos:
            G.add_node(
                f"evento_{e[0]}",
                Label=e[1][:45],
                rotulo=e[1][:45],
                tipo_no="evento",
                eixo=e[4] or "geral",
                divergencia=float(e[5] or 0.0),
                total_materias=int(e[2]),
                total_veiculos=int(e[3])
            )

        veiculos_adicionados = set()
        for a in arestas:
            ev_id, v_id, v_nome, v_tipo, sent, cb, qtd = a
            no_v = f"veiculo_{v_nome}"
            if no_v not in veiculos_adicionados:
                G.add_node(
                    no_v,
                    Label=v_nome,
                    rotulo=v_nome,
                    tipo_no="veiculo",
                    tipo_editorial=v_tipo,
                    total_materias=0,
                    total_veiculos=10
                )
                veiculos_adicionados.add(no_v)

            dist = round(abs(float(sent or 0.0)) + 0.05, 4)
            G.add_edge(
                no_v,
                f"evento_{ev_id}",
                weight=int(qtd),
                distancia_enquadramento=dist,
                sentimento=round(float(sent or 0.0), 4),
                clickbait=round(float(cb or 0.0), 4)
            )

        veiculos = [n for n, d in G.nodes(data=True) if d.get("tipo_no") == "veiculo"]
        dijkstra_caminhos = {}
        for i in range(len(veiculos)):
            for j in range(i + 1, len(veiculos)):
                v1, v2 = veiculos[i], veiculos[j]
                caminho = nx.shortest_path(G, source=v1, target=v2, weight="distancia_enquadramento")
                dist_val = nx.shortest_path_length(G, source=v1, target=v2, weight="distancia_enquadramento")
                dijkstra_caminhos[f"{v1}__{v2}"] = {
                    "origem": G.nodes[v1].get("Label"),
                    "destino": G.nodes[v2].get("Label"),
                    "distancia": round(dist_val, 4),
                    "evento_ponte": G.nodes[caminho[1]].get("Label")
                }

        bc = nx.betweenness_centrality(G, weight="distancia_enquadramento")
        eventos_bc = [(n, val) for n, val in bc.items() if G.nodes[n].get("tipo_no") == "evento"]
        top_pontes = [
            {"evento": G.nodes[no].get("Label"), "score_intermediacao": round(score, 6)}
            for no, score in sorted(eventos_bc, key=lambda x: x[1], reverse=True)[:10]
        ]

        bfs_hops = []
        for v in veiculos:
            lengths = nx.single_source_shortest_path_length(G, v)
            other_veics = [lengths[v_other] for v_other in veiculos if v_other != v]
            bfs_hops.extend(other_veics)

        analise = {
            "total_macro_eventos": len(eventos),
            "dijkstra_pares": dijkstra_caminhos,
            "top_eventos_ponte_betweenness": top_pontes,
            "diametro_bfs": max(bfs_hops) if bfs_hops else 0,
            "media_saltos_bfs": float(np.mean(bfs_hops)) if bfs_hops else 0.0
        }

        return G, analise

    def exportar_gexf(self, G: nx.Graph, nome_arquivo: str) -> str:
        caminho = self.dir_relatorios / nome_arquivo
        nx.write_gexf(G, str(caminho))
        self.logger.info(f"Grafo exportado para Gephi em: {caminho}")
        return str(caminho)

    def executar(self) -> Dict[str, Any]:
        G_bip = self.construir_grafo_bipartido()
        G_proj = self.construir_projecao_veiculos(G_bip)
        G_macro, analise_macro = self.construir_subgrafo_macro_eventos()

        caminho_bip = self.exportar_gexf(G_bip, "rede_bipartida.gexf")
        caminho_proj = self.exportar_gexf(G_proj, "rede_veiculos.gexf")
        caminho_macro = self.exportar_gexf(G_macro, "rede_macro_eventos.gexf")

        metricas = {
            "bipartido_nos": G_bip.number_of_nos() if hasattr(G_bip, "number_of_nos") else G_bip.number_of_nodes(),
            "bipartido_arestas": G_bip.number_of_edges(),
            "projecao_nos": G_proj.number_of_nodes(),
            "projecao_arestas": G_proj.number_of_edges(),
            "macro_eventos_nos": G_macro.number_of_nodes(),
            "macro_eventos_arestas": G_macro.number_of_edges(),
            "caminho_gexf_bipartido": caminho_bip,
            "caminho_gexf_projecao": caminho_proj,
            "caminho_gexf_macro": caminho_macro,
            "analise_macro": analise_macro,
            "grafo_projecao": G_proj
        }

        self.logger.info(f"Processamento de grafos concluido: {metricas}")
        return metricas
