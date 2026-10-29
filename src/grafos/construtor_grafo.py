import gc
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Tuple
import networkx as nx
import networkx.algorithms.community as nx_comm
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

        self.cor_tradicional = {"r": 27, "g": 79, "b": 114}
        self.cor_digital = {"r": 211, "g": 84, "b": 0}

        self.cores_eixos = {
            "politico": {"r": 41, "g": 128, "b": 185},
            "economico": {"r": 243, "g": 156, "b": 18},
            "ambiental": {"r": 39, "g": 174, "b": 96},
            "social": {"r": 142, "g": 68, "b": 173},
            "internacional": {"r": 192, "g": 57, "b": 43},
            "esportivo": {"r": 230, "g": 126, "b": 34},
        }

    def limpar_grafos_antigos(self) -> None:
        arquivos_antigos = [
            "rede_bipartida.gexf",
            "rede_veiculos.gexf",
            "rede_macro_eventos.gexf",
            "grafo_semantico_palavras.gexf"
        ]
        for arq in arquivos_antigos:
            p = self.dir_relatorios / arq
            if p.exists():
                p.unlink()
                self.logger.info(f"Arquivo antigo removido: {p}")

    def carregar_macro_eventos(self) -> Dict[int, Dict[str, Any]]:
        self.logger.info("Carregando 2.300 macro-eventos balanceados em 6 eixos...")
        con = self.banco.obter_conexao()
        cur = con.cursor()

        eixos_cotas = {
            "politico": 500,
            "economico": 400,
            "internacional": 400,
            "social": 400,
            "ambiental": 300,
            "esportivo": 300
        }

        eventos = {}
        for eixo, cota in eixos_cotas.items():
            cur.execute(f"""
                SELECT id, nome_evento, total_materias, total_veiculos, eixo_predominante, divergencia_sentimento
                FROM clusters_eventos
                WHERE eixo_predominante = '{eixo}' AND total_materias >= 5
                ORDER BY total_materias DESC
                LIMIT {cota}
            """)
            for r in cur.fetchall():
                titulo_limpo = r[1].replace('"', '').replace("'", "").strip()
                if len(titulo_limpo) > 42:
                    titulo_limpo = titulo_limpo[:39] + "..."
                rotulo = f"{r[4].capitalize()}: {titulo_limpo}"

                eventos[r[0]] = {
                    "id": r[0],
                    "nome": r[1],
                    "rotulo": rotulo,
                    "total_materias": r[2],
                    "total_veiculos": r[3],
                    "eixo": r[4],
                    "divergencia_sentimento": float(r[5]) if r[5] is not None else 0.0,
                    "veiculos": {},
                    "trad_count": 0,
                    "dig_count": 0,
                    "cb_scores": [],
                    "sent_scores": []
                }

        cur.execute("CREATE TEMP TABLE tmp_sel_eventos(id INTEGER PRIMARY KEY)")
        cur.executemany("INSERT INTO tmp_sel_eventos VALUES (?)", [(eid,) for eid in eventos.keys()])

        cur.execute("""
            SELECT m.evento_id, m.veiculo_id, COUNT(*), AVG(m.score_clickbait), AVG(m.sentimento_polaridade)
            FROM tmp_sel_eventos t
            JOIN materias m ON m.evento_id = t.id
            GROUP BY m.evento_id, m.veiculo_id
        """)

        for eid, vid, cnt, cb, sent in cur.fetchall():
            if eid in eventos:
                eventos[eid]["veiculos"][vid] = {
                    "count": cnt,
                    "clickbait": cb or 0.0,
                    "sentimento": sent or 0.0
                }
                if vid <= 5:
                    eventos[eid]["trad_count"] += cnt
                else:
                    eventos[eid]["dig_count"] += cnt
                if cb is not None:
                    eventos[eid]["cb_scores"].append(cb)
                if sent is not None:
                    eventos[eid]["sent_scores"].append(sent)

        con.close()

        for eid, ev in eventos.items():
            t = ev["trad_count"]
            d = ev["dig_count"]
            tot = t + d + 1e-5
            ev["polarizacao"] = (d - t) / tot
            ev["clickbait_medio"] = float(np.mean(ev["cb_scores"])) if ev["cb_scores"] else 0.0
            ev["sentimento_medio"] = float(np.mean(ev["sent_scores"])) if ev["sent_scores"] else 0.0

        self.logger.info(f"Sucesso: {len(eventos)} macro-eventos consolidados com dados de veiculos.")
        return eventos

    def construir_grafo_cascata_hierarquica(
        self,
        eventos: Dict[int, Dict[str, Any]]
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        self.logger.info("Construindo Opcao 1: Grafo Cascata Hierarquica Multinivel...")
        G = nx.Graph()
        np.random.seed(42)

        pos_veiculos_cascata = {
            "Tradicional 1": (-460.0, 520.0),
            "Tradicional 2": (-380.0, 520.0),
            "Tradicional 3": (-300.0, 520.0),
            "Tradicional 4": (-220.0, 520.0),
            "Tradicional 5": (-140.0, 520.0),
            "Digital 1": (140.0, 520.0),
            "Digital 2": (220.0, 520.0),
            "Digital 3": (300.0, 520.0),
            "Digital 4": (380.0, 520.0),
            "Digital 5": (460.0, 520.0),
        }

        for v_id in range(1, 6):
            nome = f"Tradicional {v_id}"
            x, y = pos_veiculos_cascata[nome]
            G.add_node(
                nome,
                label=nome,
                rotulo=nome,
                tipo_no="veiculo",
                grupo_editorial="Tradicional",
                eixo="Imprensa Tradicional",
                polarizacao_editorial=-1.0,
                divergencia_sentimento=0.0,
                total_materias=0,
                total_veiculos=1,
                size=60.0,
                viz={"size": 60.0, "color": self.cor_tradicional, "position": {"x": x, "y": y, "z": 0.0}}
            )

        for v_id in range(1, 6):
            nome = f"Digital {v_id}"
            x, y = pos_veiculos_cascata[nome]
            G.add_node(
                nome,
                label=nome,
                rotulo=nome,
                tipo_no="veiculo",
                grupo_editorial="Digital Nativo",
                eixo="Nativos Digitais",
                polarizacao_editorial=1.0,
                divergencia_sentimento=0.0,
                total_materias=0,
                total_veiculos=1,
                size=60.0,
                viz={"size": 60.0, "color": self.cor_digital, "position": {"x": x, "y": y, "z": 0.0}}
            )

        ilhas_centros = {
            "politico": (-340.0, 100.0, 125.0, self.cores_eixos["politico"]),
            "economico": (0.0, 130.0, 110.0, self.cores_eixos["economico"]),
            "internacional": (340.0, 100.0, 110.0, self.cores_eixos["internacional"]),
            "social": (-280.0, -220.0, 110.0, self.cores_eixos["social"]),
            "ambiental": (0.0, -240.0, 95.0, self.cores_eixos["ambiental"]),
            "esportivo": (280.0, -220.0, 95.0, self.cores_eixos["esportivo"]),
        }

        max_mat = max(ev["total_materias"] for ev in eventos.values()) if eventos else 1
        eventos_por_eixo = defaultdict(list)

        for eid, ev in eventos.items():
            eventos_por_eixo[ev["eixo"]].append(eid)
            cx, cy, r_ilha, cor_ilha = ilhas_centros.get(
                ev["eixo"], (0.0, 0.0, 100.0, {"r": 120, "g": 120, "b": 120})
            )

            u = np.random.uniform(0.04, 1.0)
            theta = np.random.uniform(0, 2 * np.pi)
            raio_no = r_ilha * np.sqrt(u)

            pol = ev["polarizacao"]
            px = cx + raio_no * np.cos(theta) + pol * (r_ilha * 0.35)
            py = cy + raio_no * np.sin(theta)

            tam = 8.5 + 16.5 * (np.log1p(ev["total_materias"]) / np.log1p(max_mat))
            str_id = str(eid)

            G.add_node(
                str_id,
                label=ev["rotulo"],
                rotulo=ev["rotulo"],
                tipo_no="evento",
                grupo_editorial="Evento Factual",
                eixo=ev["eixo"],
                polarizacao_editorial=round(float(pol), 4),
                divergencia_sentimento=round(float(ev["divergencia_sentimento"]), 4),
                total_materias=int(ev["total_materias"]),
                total_veiculos=int(ev["total_veiculos"]),
                size=round(float(tam), 2),
                viz={
                    "size": round(float(tam), 2),
                    "color": cor_ilha,
                    "position": {"x": round(float(px), 2), "y": round(float(py), 2), "z": 0.0}
                }
            )

        for eid, ev in eventos.items():
            str_id = str(eid)
            tot_evento = sum(v["count"] for v in ev["veiculos"].values())
            for vid, v_data in ev["veiculos"].items():
                cnt = v_data["count"]
                prop = cnt / (tot_evento + 1e-5)
                if prop >= 0.10 or len(ev["veiculos"]) <= 2:
                    v_nome = f"Tradicional {vid}" if vid <= 5 else f"Digital {vid - 5}"
                    w_norm = 0.4 + 1.2 * min(1.0, cnt / 15.0)
                    dist = 1.0 / (1.0 + np.log1p(cnt))
                    if vid <= 5:
                        cor_aresta = {"r": 100, "g": 149, "b": 237, "a": 0.28}
                    else:
                        cor_aresta = {"r": 255, "g": 140, "b": 0, "a": 0.28}

                    G.add_edge(
                        v_nome,
                        str_id,
                        weight=round(float(w_norm), 3),
                        distancia=round(float(dist), 4),
                        tipo_aresta="cobertura_cascata",
                        viz={"color": cor_aresta}
                    )

        for eixo, eids in eventos_por_eixo.items():
            sub_eids = sorted(eids, key=lambda i: eventos[i]["polarizacao"])
            for idx in range(len(sub_eids) - 1):
                if idx % 2 == 0:
                    id1 = str(sub_eids[idx])
                    id2 = str(sub_eids[idx + 1])
                    diff_pol = abs(eventos[sub_eids[idx]]["polarizacao"] - eventos[sub_eids[idx + 1]]["polarizacao"])
                    w_intra = 0.35 + 0.3 * max(0.0, 1.0 - diff_pol)
                    dist_intra = 1.0 / (1.0 + np.log1p(w_intra))
                    G.add_edge(
                        id1,
                        id2,
                        weight=round(float(w_intra), 3),
                        distancia=round(float(dist_intra), 4),
                        tipo_aresta="proximidade_intra_ilha",
                        viz={"color": {"r": 200, "g": 200, "b": 210, "a": 0.15}}
                    )

        metricas = self._calcular_metricas_grafo(G, "cascata_hierarquica")
        return G, metricas

    def construir_grafo_planisferio_polarizacao(
        self,
        eventos: Dict[int, Dict[str, Any]]
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        self.logger.info("Construindo Opcao 2: Grafo Planisferio Cosmico de Polarizacao...")
        G = nx.Graph()
        np.random.seed(42)

        r_orb = 460.0
        angulos_trad = [135.0, 157.5, 180.0, 202.5, 225.0]
        for idx, deg in enumerate(angulos_trad):
            nome = f"Tradicional {idx + 1}"
            rad = np.radians(deg)
            x = r_orb * np.cos(rad)
            y = r_orb * np.sin(rad)
            G.add_node(
                nome,
                label=nome,
                rotulo=nome,
                tipo_no="veiculo",
                grupo_editorial="Tradicional",
                eixo="Imprensa Tradicional",
                polarizacao_editorial=-1.0,
                divergencia_sentimento=0.0,
                total_materias=0,
                total_veiculos=1,
                size=65.0,
                viz={"size": 65.0, "color": self.cor_tradicional, "position": {"x": round(float(x), 2), "y": round(float(y), 2), "z": 0.0}}
            )

        angulos_dig = [-45.0, -22.5, 0.0, 22.5, 45.0]
        for idx, deg in enumerate(angulos_dig):
            nome = f"Digital {idx + 1}"
            rad = np.radians(deg)
            x = r_orb * np.cos(rad)
            y = r_orb * np.sin(rad)
            G.add_node(
                nome,
                label=nome,
                rotulo=nome,
                tipo_no="veiculo",
                grupo_editorial="Digital Nativo",
                eixo="Nativos Digitais",
                polarizacao_editorial=1.0,
                divergencia_sentimento=0.0,
                total_materias=0,
                total_veiculos=1,
                size=65.0,
                viz={"size": 65.0, "color": self.cor_digital, "position": {"x": round(float(x), 2), "y": round(float(y), 2), "z": 0.0}}
            )

        latitudes_eixo = {
            "internacional": 230.0,
            "politico": 115.0,
            "economico": 0.0,
            "social": -110.0,
            "ambiental": -200.0,
            "esportivo": -270.0
        }

        max_mat = max(ev["total_materias"] for ev in eventos.values()) if eventos else 1
        eventos_equador = []

        for eid, ev in eventos.items():
            pol = ev["polarizacao"]
            str_id = str(eid)

            jx = np.random.normal(0, 14.0)
            px = pol * 340.0 + jx
            px = float(np.clip(px, -370.0, 370.0))

            envelope = np.sqrt(max(0.12, 1.0 - (px / 380.0) ** 2))
            y_base = latitudes_eixo.get(ev["eixo"], 0.0)
            jy = np.random.uniform(-25.0, 25.0)
            py = float((y_base + jy) * envelope)

            if pol < -0.2:
                t = min(1.0, abs(pol))
                cor_no = {
                    "r": int(27 + (52 - 27) * (1.0 - t)),
                    "g": int(79 + (152 - 79) * (1.0 - t)),
                    "b": int(114 + (219 - 114) * (1.0 - t))
                }
            elif pol > 0.2:
                t = min(1.0, pol)
                cor_no = {
                    "r": int(243 + (231 - 243) * t),
                    "g": int(156 + (76 - 156) * t),
                    "b": int(18 + (60 - 18) * t)
                }
            else:
                cor_no = {"r": 142, "g": 68, "b": 173}
                eventos_equador.append(eid)

            tam = 8.5 + 16.5 * (np.log1p(ev["total_materias"]) / np.log1p(max_mat))

            G.add_node(
                str_id,
                label=ev["rotulo"],
                rotulo=ev["rotulo"],
                tipo_no="evento",
                grupo_editorial="Evento Factual",
                eixo=ev["eixo"],
                polarizacao_editorial=round(float(pol), 4),
                divergencia_sentimento=round(float(ev["divergencia_sentimento"]), 4),
                total_materias=int(ev["total_materias"]),
                total_veiculos=int(ev["total_veiculos"]),
                size=round(float(tam), 2),
                viz={
                    "size": round(float(tam), 2),
                    "color": cor_no,
                    "position": {"x": round(float(px), 2), "y": round(float(py), 2), "z": 0.0}
                }
            )

        for eid, ev in eventos.items():
            str_id = str(eid)
            tot_evento = sum(v["count"] for v in ev["veiculos"].values())
            for vid, v_data in ev["veiculos"].items():
                cnt = v_data["count"]
                prop = cnt / (tot_evento + 1e-5)
                if prop >= 0.10 or len(ev["veiculos"]) <= 2:
                    v_nome = f"Tradicional {vid}" if vid <= 5 else f"Digital {vid - 5}"
                    w_norm = 0.4 + 1.2 * min(1.0, cnt / 15.0)
                    dist = 1.0 / (1.0 + np.log1p(cnt))
                    if vid <= 5:
                        cor_aresta = {"r": 100, "g": 149, "b": 237, "a": 0.25}
                    else:
                        cor_aresta = {"r": 255, "g": 140, "b": 0, "a": 0.25}

                    G.add_edge(
                        v_nome,
                        str_id,
                        weight=round(float(w_norm), 3),
                        distancia=round(float(dist), 4),
                        tipo_aresta="cobertura_polar",
                        viz={"color": cor_aresta}
                    )

        sub_eq = sorted(eventos_equador, key=lambda i: eventos[i]["total_materias"], reverse=True)[:350]
        for i in range(0, len(sub_eq) - 1, 2):
            id1 = str(sub_eq[i])
            id2 = str(sub_eq[i + 1])
            G.add_edge(
                id1,
                id2,
                weight=0.45,
                distancia=0.6,
                tipo_aresta="consenso_equatorial",
                viz={"color": {"r": 180, "g": 180, "b": 220, "a": 0.18}}
            )

        metricas = self._calcular_metricas_grafo(G, "planisferio_polarizacao")
        return G, metricas

    def construir_grafo_projecao_veiculos(
        self,
        eventos: Dict[int, Dict[str, Any]]
    ) -> nx.Graph:
        G_proj = nx.Graph()
        veiculos_trad = [f"Tradicional {i}" for i in range(1, 6)]
        veiculos_dig = [f"Digital {i}" for i in range(1, 6)]
        todos_veiculos = veiculos_trad + veiculos_dig

        for v in veiculos_trad:
            G_proj.add_node(v, rotulo=v, tipo_editorial="tradicional")
        for v in veiculos_dig:
            G_proj.add_node(v, rotulo=v, tipo_editorial="digital")

        vetores = defaultdict(lambda: np.zeros(len(eventos), dtype=np.float32))
        for idx_ev, (eid, ev) in enumerate(eventos.items()):
            for vid, v_data in ev["veiculos"].items():
                nome = f"Tradicional {vid}" if vid <= 5 else f"Digital {vid - 5}"
                vetores[nome][idx_ev] = float(v_data["count"])

        for i in range(len(todos_veiculos)):
            for j in range(i + 1, len(todos_veiculos)):
                v1, v2 = todos_veiculos[i], todos_veiculos[j]
                vec1, vec2 = vetores[v1], vetores[v2]
                norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    cos_sim = float(np.dot(vec1, vec2) / (norm1 * norm2))
                    if cos_sim >= 0.15:
                        G_proj.add_edge(v1, v2, weight=round(cos_sim, 4))

        return G_proj

    def _calcular_metricas_grafo(self, G: nx.Graph, modelo_nome: str) -> Dict[str, Any]:
        self.logger.info(f"Calculando metricas analiticas para {modelo_nome}...")

        comunidades = nx_comm.louvain_communities(G, weight="weight", resolution=1.0)
        mod_q = nx_comm.modularity(G, comunidades, weight="weight")

        for idx_c, comm in enumerate(comunidades):
            for no in comm:
                G.nodes[no]["modularity_class"] = idx_c

        veics_trad = [f"Tradicional {i}" for i in range(1, 6)]
        veics_dig = [f"Digital {i}" for i in range(1, 6)]

        d_trad = []
        d_dig = []
        d_cross = []
        caminhos_exemplares = []

        for i in range(len(veics_trad)):
            for j in range(i + 1, len(veics_trad)):
                v1, v2 = veics_trad[i], veics_trad[j]
                if nx.has_path(G, v1, v2):
                    d_trad.append(nx.shortest_path_length(G, v1, v2, weight="distancia"))

        for i in range(len(veics_dig)):
            for j in range(i + 1, len(veics_dig)):
                v1, v2 = veics_dig[i], veics_dig[j]
                if nx.has_path(G, v1, v2):
                    d_dig.append(nx.shortest_path_length(G, v1, v2, weight="distancia"))

        for v1 in veics_trad:
            for v2 in veics_dig:
                if nx.has_path(G, v1, v2):
                    d = nx.shortest_path_length(G, v1, v2, weight="distancia")
                    d_cross.append(d)
                    if len(caminhos_exemplares) < 5:
                        caminho = nx.shortest_path(G, v1, v2, weight="distancia")
                        if len(caminho) >= 3:
                            ponte_id = caminho[1]
                            rotulo_ponte = G.nodes[ponte_id].get("rotulo", ponte_id)
                            caminhos_exemplares.append({
                                "origem": v1,
                                "destino": v2,
                                "distancia": round(float(d), 4),
                                "evento_ponte": rotulo_ponte
                            })

        mean_trad = float(np.mean(d_trad)) if d_trad else 0.0
        mean_dig = float(np.mean(d_dig)) if d_dig else 0.0
        mean_cross = float(np.mean(d_cross)) if d_cross else 0.0
        razao_segregacao = mean_cross / (0.5 * (mean_trad + mean_dig) + 1e-6)

        bc = nx.betweenness_centrality(G, k=250, weight="distancia")
        eventos_bc = [(n, val) for n, val in bc.items() if G.nodes[n].get("tipo_no") == "evento"]
        top_pontes = [
            {
                "evento_id": no,
                "rotulo": G.nodes[no].get("rotulo"),
                "eixo": G.nodes[no].get("eixo"),
                "score": round(float(score), 6)
            }
            for no, score in sorted(eventos_bc, key=lambda x: x[1], reverse=True)[:10]
        ]

        bfs_saltos = []
        for v in veics_trad + veics_dig:
            lengths = nx.single_source_shortest_path_length(G, v)
            for v_alvo in veics_trad + veics_dig:
                if v != v_alvo and v_alvo in lengths:
                    bfs_saltos.append(lengths[v_alvo])

        resultado = {
            "modelo": modelo_nome,
            "total_nos": G.number_of_nodes(),
            "total_arestas": G.number_of_edges(),
            "total_comunidades_louvain": len(comunidades),
            "modularidade_q": round(float(mod_q), 4),
            "distancia_media_dijkstra_tradicional": round(mean_trad, 4),
            "distancia_media_dijkstra_digital": round(mean_dig, 4),
            "distancia_media_dijkstra_cruzada": round(mean_cross, 4),
            "razao_segregacao_geodesica": round(float(razao_segregacao), 4),
            "caminhos_geodesicos_exemplares": caminhos_exemplares,
            "top_eventos_ponte_betweenness": top_pontes,
            "diametro_bfs_entre_veiculos": max(bfs_saltos) if bfs_saltos else 0,
            "media_saltos_bfs": round(float(np.mean(bfs_saltos)), 2) if bfs_saltos else 0.0
        }
        return resultado

    def exportar_gexf(self, G: nx.Graph, nome_arquivo: str) -> str:
        caminho = self.dir_relatorios / nome_arquivo
        nx.write_gexf(G, str(caminho))
        self.logger.info(f"Grafo exportado para Gephi em: {caminho}")
        return str(caminho)

    def executar(self) -> Dict[str, Any]:
        self.limpar_grafos_antigos()
        eventos = self.carregar_macro_eventos()

        G_cascata, analise_cascata = self.construir_grafo_cascata_hierarquica(eventos)
        caminho_cascata = self.exportar_gexf(G_cascata, "grafo_cascata_hierarquica.gexf")

        G_planisferio, analise_planisferio = self.construir_grafo_planisferio_polarizacao(eventos)
        caminho_planisferio = self.exportar_gexf(G_planisferio, "grafo_planisferio_polarizacao.gexf")

        G_proj = self.construir_grafo_projecao_veiculos(eventos)

        self.banco.salvar_resultado_estatistico(
            pergunta="grafo_cascata",
            chave="analise_hierarquica_multinivel",
            valor=analise_cascata.get("modularidade_q", 0.0),
            detalhes=analise_cascata
        )

        self.banco.salvar_resultado_estatistico(
            pergunta="grafo_planisferio",
            chave="analise_planisferio_polarizacao",
            valor=analise_planisferio.get("modularidade_q", 0.0),
            detalhes=analise_planisferio
        )

        metricas = {
            "cascata_nos": G_cascata.number_of_nodes(),
            "cascata_arestas": G_cascata.number_of_edges(),
            "caminho_cascata": caminho_cascata,
            "analise_cascata": analise_cascata,
            "planisferio_nos": G_planisferio.number_of_nodes(),
            "planisferio_arestas": G_planisferio.number_of_edges(),
            "caminho_planisferio": caminho_planisferio,
            "analise_planisferio": analise_planisferio,
            "dir_grafos": str(self.dir_relatorios),
            "grafo_projecao": G_proj,
            "bipartido_nos": G_cascata.number_of_nodes(),
            "bipartido_arestas": G_cascata.number_of_edges(),
            "projecao_nos": G_proj.number_of_nodes(),
            "projecao_arestas": G_proj.number_of_edges(),
            "caminho_gexf_projecao": caminho_cascata
        }

        self.logger.info("Construcao de ambos os grafos concluida com exito.")
        return metricas
