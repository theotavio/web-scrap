import csv
import gc
import io
from pathlib import Path
import re
import zipfile
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import plotly.graph_objects as go
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.banco import BancoDados


class ConstrutorGrafos:
    def __init__(self, banco: BancoDados):
        self.banco = banco
        self.config = obter_configuracao()
        self.logger = obter_logger("grafos")
        self.dir_relatorios = Path(self.config.obter("geral.diretorio_relatorios", "resultados")) / "grafos" / "arquivos"
        self.dir_relatorios.mkdir(parents=True, exist_ok=True)

        self.cor_tradicional = {"r": 27, "g": 79, "b": 114}
        self.cor_digital = {"r": 211, "g": 84, "b": 0}

        self.cores_eixos = {
            "politico": {"r": 139, "g": 92, "b": 246},
            "economico": {"r": 234, "g": 179, "b": 8},
            "social": {"r": 236, "g": 72, "b": 153},
            "ambiental": {"r": 16, "g": 185, "b": 129},
            "internacional": {"r": 6, "g": 182, "b": 212}
        }

    def limpar_grafos_antigos(self) -> None:
        arquivos_antigos = [
            "rede_bipartida.gexf",
            "rede_veiculos.gexf",
            "rede_macro_eventos.gexf",
            "grafo_cascata_hierarquica.gexf",
            "grafo_similaridade_veiculos.gexf"
        ]
        for arq in arquivos_antigos:
            p = self.dir_relatorios / arq
            if p.exists():
                p.unlink()
                self.logger.info(f"Arquivo antigo removido: {p}")

    def carregar_macro_eventos(self) -> Dict[int, Dict[str, Any]]:
        self.logger.info("Carregando macro-eventos balanceados nos 5 eixos oficiais...")
        con = self.banco.obter_conexao()
        cur = con.cursor()

        eixos_cotas = {
            "politico": 6000,
            "economico": 4000,
            "internacional": 3500,
            "social": 3000,
            "ambiental": 1800
        }

        eventos = {}
        for eixo, cota in eixos_cotas.items():
            cur.execute(f"""
                SELECT id, nome_evento, total_materias, total_veiculos, eixo_predominante, divergencia_sentimento
                FROM clusters_eventos
                WHERE eixo_predominante = '{eixo}' AND total_materias >= 3
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
            CROSS JOIN materias m ON m.evento_id = t.id
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

    def construir_grafo_planisferio_polarizacao(
        self,
        eventos: Dict[int, Dict[str, Any]]
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        self.logger.info("Construindo Grafo Planisferio Cosmico de Polarizacao (20.000 eventos)...")
        G = nx.Graph()
        np.random.seed(42)

        r_orb = 680.0
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
                size=80.0,
                viz={"size": 80.0, "color": self.cor_tradicional, "position": {"x": round(float(x), 2), "y": round(float(y), 2), "z": 0.0}}
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
                size=80.0,
                viz={"size": 80.0, "color": self.cor_digital, "position": {"x": round(float(x), 2), "y": round(float(y), 2), "z": 0.0}}
            )

        latitudes_eixo = {
            "internacional": 330.0,
            "politico": 165.0,
            "economico": 0.0,
            "social": -160.0,
            "ambiental": -280.0
        }

        max_mat = max(ev["total_materias"] for ev in eventos.values()) if eventos else 1
        eventos_equador = []

        for eid, ev in eventos.items():
            pol = ev["polarizacao"]
            str_id = str(eid)

            jx = np.random.normal(0, 16.0)
            px = pol * 490.0 + jx
            px = float(np.clip(px, -530.0, 530.0))

            envelope = np.sqrt(max(0.10, 1.0 - (px / 550.0) ** 2))
            y_base = latitudes_eixo.get(ev["eixo"], 0.0)
            jy = np.random.uniform(-30.0, 30.0)
            py = float((y_base + jy) * envelope)

            if pol < -0.15:
                t = min(1.0, abs(pol))
                cor_no = {
                    "r": int(27 + (52 - 27) * (1.0 - t)),
                    "g": int(79 + (152 - 79) * (1.0 - t)),
                    "b": int(114 + (219 - 114) * (1.0 - t))
                }
            elif pol > 0.15:
                t = min(1.0, pol)
                cor_no = {
                    "r": int(243 + (231 - 243) * t),
                    "g": int(156 + (76 - 156) * t),
                    "b": int(18 + (60 - 18) * t)
                }
            else:
                cor_no = {"r": 142, "g": 68, "b": 173}
                eventos_equador.append(eid)

            tam = 3.0 + 8.5 * (np.log1p(ev["total_materias"]) / np.log1p(max_mat))

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
                if prop >= 0.08 or len(ev["veiculos"]) <= 2:
                    v_nome = f"Tradicional {vid}" if vid <= 5 else f"Digital {vid - 5}"
                    w_norm = 0.3 + 1.2 * min(1.0, cnt / 15.0)
                    dist = 1.0 / (1.0 + np.log1p(cnt))
                    if vid <= 5:
                        cor_aresta = {"r": 100, "g": 149, "b": 237, "a": 0.20}
                    else:
                        cor_aresta = {"r": 255, "g": 140, "b": 0, "a": 0.20}

                    G.add_edge(
                        v_nome,
                        str_id,
                        weight=round(float(w_norm), 3),
                        distancia=round(float(dist), 4),
                        tipo_aresta="cobertura_polar",
                        viz={"color": cor_aresta}
                    )

        sub_eq = sorted(eventos_equador, key=lambda i: eventos[i]["total_materias"], reverse=True)[:800]
        for i in range(0, len(sub_eq) - 1, 2):
            id1 = str(sub_eq[i])
            id2 = str(sub_eq[i + 1])
            G.add_edge(
                id1,
                id2,
                weight=0.4,
                distancia=0.65,
                tipo_aresta="consenso_equatorial",
                viz={"color": {"r": 180, "g": 180, "b": 220, "a": 0.15}}
            )

        metricas = self._calcular_metricas_grafo(G, "planisferio_polarizacao")
        return G, metricas

    def _calcular_metricas_grafo(self, G: nx.Graph, modelo_nome: str) -> Dict[str, Any]:
        self.logger.info(f"Calculando metricas analiticas para {modelo_nome}...")

        comunidades = nx_comm.louvain_communities(G, weight="weight", resolution=1.0, seed=42)
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

        bc = nx.betweenness_centrality(G, k=100, weight="distancia", seed=42)
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

        G_planisferio, analise_planisferio = self.construir_grafo_planisferio_polarizacao(eventos)
        caminho_planisferio = self.exportar_gexf(G_planisferio, "grafo_planisferio_polarizacao.gexf")

        self.banco.salvar_resultado_estatistico(
            pergunta="grafo_planisferio",
            chave="analise_planisferio_polarizacao",
            valor=analise_planisferio.get("modularidade_q", 0.0),
            detalhes=analise_planisferio
        )

        metricas = {
            "planisferio_nos": G_planisferio.number_of_nodes(),
            "planisferio_arestas": G_planisferio.number_of_edges(),
            "caminho_planisferio": caminho_planisferio,
            "analise_planisferio": analise_planisferio,
            "dir_grafos": str(self.dir_relatorios)
        }

        self.logger.info("Construcao do grafo do planisferio concluida com exito.")
        return metricas

    def _adicionar_no_planisferio(self, G: nx.Graph, row: Dict[str, str]) -> None:
        nid = row["id"]
        lbl = row["label"]
        t_no = row["tipo_no"]
        grupo = row["grupo_editorial"]
        eixo = row.get("eixo_tematico", "")
        pol = float(row.get("polarizacao_editorial", 0.0))
        div = float(row.get("divergencia_sentimento", 0.0))
        tot_mat = int(row.get("total_materias", 1))
        tot_veic = int(row.get("total_veiculos", 1))
        mod_class = int(row.get("modularity_class", 0))
        px = float(row.get("pos_x", 0.0))
        py = float(row.get("pos_y", 0.0))

        if t_no == "veiculo":
            cor = self.cor_tradicional if "Tradicional" in grupo else self.cor_digital
            tam = float(45.0)
        else:
            cor = self.cores_eixos.get(eixo.lower(), {"r": 100, "g": 100, "b": 100})
            tam = float(5.0 + 3.0 * min(8.0, float(np.log1p(tot_mat))))

        tam_val = float(round(tam, 2))
        G.add_node(
            nid,
            label=str(lbl),
            tipo_no=str(t_no),
            grupo_editorial=str(grupo),
            eixo_tematico=str(eixo),
            polarizacao_editorial=float(pol),
            divergencia_sentimento=float(div),
            total_materias=int(tot_mat),
            total_veiculos=int(tot_veic),
            modularity_class=int(mod_class),
            size=tam_val,
            viz={"size": tam_val, "color": cor, "position": {"x": float(px), "y": float(py), "z": 0.0}}
        )

    def executar_a_partir_de_pasta(self, dir_dados: str) -> Dict[str, Any]:
        p_dados = Path(dir_dados) if dir_dados else Path("resultados/grafos/arquivos")
        if not p_dados.exists():
            raise FileNotFoundError(f"Diretório informado não existe: {dir_dados}")
        p_zip = p_dados / "dados_grafos_gephi.zip"
        p_nos = p_dados / "01_grafo_planisferio_nos_polarizacao.csv"
        p_arestas = p_dados / "01_grafo_planisferio_arestas_orbitais.csv"

        self.logger.info(f"Construindo grafo do Planisfério a partir do pacote em {dir_dados}...")
        G = nx.Graph()

        if p_zip.exists():
            with zipfile.ZipFile(p_zip, "r") as zf:
                if "01_grafo_planisferio_nos_polarizacao.csv" not in zf.namelist() or "01_grafo_planisferio_arestas_orbitais.csv" not in zf.namelist():
                    raise FileNotFoundError(f"Arquivos do Planisfério não encontrados dentro de {p_zip}")
                with zf.open("01_grafo_planisferio_nos_polarizacao.csv") as f_nos:
                    reader = csv.DictReader(io.TextIOWrapper(f_nos, encoding="utf-8"))
                    for row in reader:
                        self._adicionar_no_planisferio(G, row)
                with zf.open("01_grafo_planisferio_arestas_orbitais.csv") as f_ar:
                    reader = csv.DictReader(io.TextIOWrapper(f_ar, encoding="utf-8"))
                    for row in reader:
                        G.add_edge(row["source"], row["target"], weight=float(row.get("weight", 1.0)), distancia=float(row.get("distancia", 1.0)), tipo_aresta=row.get("tipo_aresta", "cobertura"))
        elif p_nos.exists() and p_arestas.exists():
            with open(p_nos, "r", encoding="utf-8") as f_nos:
                reader = csv.DictReader(f_nos)
                for row in reader:
                    self._adicionar_no_planisferio(G, row)
            with open(p_arestas, "r", encoding="utf-8") as f_ar:
                reader = csv.DictReader(f_ar)
                for row in reader:
                    G.add_edge(row["source"], row["target"], weight=float(row.get("weight", 1.0)), distancia=float(row.get("distancia", 1.0)), tipo_aresta=row.get("tipo_aresta", "cobertura"))
        else:
            raise FileNotFoundError(f"Arquivos do Planisfério (CSVs ou dados_grafos_gephi.zip) não encontrados em {dir_dados}")

        caminho_planisferio = self.exportar_gexf(G, "grafo_planisferio_polarizacao.gexf")
        return {
            "planisferio_nos": G.number_of_nodes(),
            "planisferio_arestas": G.number_of_edges(),
            "caminho_planisferio": caminho_planisferio,
            "dir_grafos": str(self.dir_relatorios)
        }

    executar_a_partir_do_zenodo = executar_a_partir_de_pasta


class ConstrutorGrafoSemantico:
    def __init__(self, banco: BancoDados):
        self.banco = banco
        self.config = obter_configuracao()
        self.logger = obter_logger("grafos")
        self.dir_grafos = Path(self.config.obter("geral.diretorio_relatorios", "resultados")) / "grafos" / "arquivos"
        self.dir_grafos.mkdir(parents=True, exist_ok=True)
        self.dir_graficos = Path(self.config.obter("geral.diretorio_relatorios", "resultados")) / "graficos"
        self.dir_graficos.mkdir(parents=True, exist_ok=True)

        self.stopwords = {
            "para", "com", "nao", "uma", "mais", "como", "ele", "seu", "sua", "quando", "muito", "tambem",
            "pelo", "pela", "depois", "mesmo", "seus", "quem", "esse", "eles", "voce", "essa", "numa",
            "pelos", "elas", "qual", "deles", "essas", "esses", "pelas", "este", "dele", "voces", "minhas",
            "nossa", "nossos", "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela",
            "aqueles", "aquelas", "isto", "aquilo", "estou", "estamos", "estao", "estive", "esteve",
            "estivemos", "estiveram", "estava", "estavamos", "estavam", "estivera", "esteja", "estejam",
            "estivesse", "houve", "houveram", "haja", "hajam", "houvesse", "houver", "somos", "foram",
            "sejam", "fossem", "seremos", "serao", "seria", "seriam", "tenho", "temos", "tinham", "teve",
            "tiveram", "tenham", "tivessem", "tiver", "sobre", "anos", "segundo", "pode", "dias", "novo",
            "nova", "novos", "novas", "primeiro", "primeira", "contra", "brasil", "brasileiro", "brasileira",
            "brasileiros", "brasileiras", "país", "pais", "governo", "veja", "tudo", "onde", "porque",
            "saiba", "entenda", "foto", "fotos", "vídeo", "vídeos", "leia", "confira", "tempo", "minuto",
            "minutos", "hora", "horas", "outro", "outra", "outros", "outras", "hoje", "ontem", "nesta",
            "neste", "disse", "deve", "fazer", "fala", "quer", "caso", "parte", "durante", "frente",
            "forma", "ponto", "grupo", "número", "ainda", "maior", "menor", "melhor", "pior", "grande",
            "grandes", "pequeno", "pequenos", "alta", "queda", "baixa", "deixa", "morre", "mortes", "morto",
            "morta", "homem", "mulher", "mulheres", "pessoas", "geral", "nacional", "brasília", "paulo",
            "janeiro", "folha", "estadao", "globo", "gazeta", "correio", "metrópoles", "metropoles",
            "poder360", "antagonista", "brasil247", "nexo", "após", "além", "sendo", "terá",
            "será", "está", "estão", "havia", "desde", "pouco", "muitos", "muitas", "feira", "desta",
            "deste", "segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo", "semana",
            "mês", "meses", "aponta", "mostra", "passa", "chega", "segue", "volta", "começa", "afirma",
            "pede", "vê", "revela", "traz", "diz", "dizer", "levar", "sair", "dar", "fica", "ter", "vir",
            "ver", "vai", "vão", "ia", "virou", "ficou", "deu", "foi", "era", "são", "tem", "entre",
            "assim", "cada", "tanto", "quanto", "sabe", "tinha", "vamos", "então", "aqui", "tudo",
            "dizendo", "querem", "discutem", "fazem", "tenham", "podem", "devem", "sendo", "sobre", "sob",
            "veículo", "veiculos", "notícia", "notícias", "artigo", "editorial", "texto", "assunto"
        }

        self.cores_eixos = {
            "politico": {"hex": "#8E44AD", "r": 142, "g": 68, "b": 173, "nome": "Político", "class": 0},
            "economico": {"hex": "#D97706", "r": 217, "g": 119, "b": 6, "nome": "Econômico", "class": 1},
            "internacional": {"hex": "#1D4ED8", "r": 29, "g": 78, "b": 216, "nome": "Internacional", "class": 2},
            "social": {"hex": "#DB2777", "r": 219, "g": 39, "b": 119, "nome": "Social / Saúde", "class": 3},
            "ambiental": {"hex": "#15803D", "r": 21, "g": 128, "b": 61, "nome": "Meio Ambiente", "class": 4}
        }

    def _obter_eventos_multi_veiculo(self, limite: int = 60000) -> List[Tuple[str, str, int, float]]:
        con = self.banco.obter_conexao()
        cur = con.cursor()
        cur.execute(f"""
            SELECT nome_evento, eixo_predominante, total_materias, divergencia_sentimento
            FROM clusters_eventos
            WHERE total_materias >= 5 AND total_veiculos >= 2
            ORDER BY total_materias DESC
            LIMIT {limite}
        """)
        linhas = [(r[0], r[1] or "politico", int(r[2]), float(r[3] or 0.0)) for r in cur.fetchall()]
        con.close()
        return linhas

    def _normalizar_tokens_titulo(self, texto: str) -> List[str]:
        t = texto.lower()
        t = re.sub(r"\blava\s+jato\b", "lava_jato", t)
        t = re.sub(r"\bbanco\s+central\b", "banco_central", t)
        t = re.sub(r"\btaxa\s+selic\b", "taxa_selic", t)
        t = re.sub(r"\breforma\s+tribut[aá]ria\b", "reforma_tributaria", t)
        t = re.sub(r"\bteto\s+de\s+gastos\b", "teto_gastos", t)
        t = re.sub(r"\bor[cç]amento\s+secreto\b", "orcamento_secreto", t)
        t = re.sub(r"\bmarco\s+temporal\b", "marco_temporal", t)
        t = re.sub(r"\baux[ií]lio\s+emergencial\b", "auxilio_emergencial", t)
        t = re.sub(r"\bbolsa\s+fam[ií]lia\b", "bolsa_familia", t)
        t = re.sub(r"\bcovid[-_ ]?19\b", "covid", t)
        t = re.sub(r"\bcoronav[ií]rus\b", "covid", t)
        tokens = re.findall(r"\b[a-záéíóúãõâêîôûç_]{4,}\b", t)
        tokens_filtrados = [tok for tok in tokens if tok not in self.stopwords and len(tok) >= 4]
        return tokens_filtrados

    def _formatar_rotulo_palavra(self, token: str) -> str:
        mapa_especial = {
            "lava_jato": "Lava Jato",
            "banco_central": "Banco Central",
            "taxa_selic": "Taxa Selic",
            "reforma_tributaria": "Reforma Tributária",
            "teto_gastos": "Teto de Gastos",
            "orcamento_secreto": "Orçamento Secreto",
            "marco_temporal": "Marco Temporal",
            "auxilio_emergencial": "Auxílio Emergencial",
            "bolsa_familia": "Bolsa Família",
            "covid": "COVID-19",
            "stf": "STF",
            "cpi": "CPI",
            "pib": "PIB",
            "eua": "EUA",
            "onu": "ONU",
            "brics": "BRICS",
            "tse": "TSE",
            "mpf": "MPF",
            "anvisa": "Anvisa",
            "mec": "MEC",
            "ibama": "Ibama",
            "cbf": "CBF",
            "sus": "SUS",
            "copom": "Copom"
        }
        if token in mapa_especial:
            return mapa_especial[token]
        return token.replace("_", " ").capitalize()

    def construir_grafo(
        self,
        palavras_por_eixo: int = 65,
        min_coocorrencia: int = 4
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        self.logger.info("Minerando corpus para Grafo Semântico de Palavras...")
        eventos = self._obter_eventos_multi_veiculo(limite=60000)

        eixos_vocab = defaultdict(lambda: Counter())
        palavra_stats = defaultdict(lambda: {"freq": 0, "divergencias": [], "eixo": "politico"})

        eventos_tokens = []
        for nome, eixo, tot, div in eventos:
            toks = self._normalizar_tokens_titulo(nome)
            eventos_tokens.append((toks, eixo, div))
            eixos_vocab[eixo].update(toks)
            for t in toks:
                palavra_stats[t]["freq"] += 1
                palavra_stats[t]["divergencias"].append(div)
                palavra_stats[t]["eixo"] = eixo

        termos_selecionados: Set[str] = set()
        for eixo, cnt in eixos_vocab.items():
            for t, _ in cnt.most_common(palavras_por_eixo):
                termos_selecionados.add(t)

        coocorrencias: Counter = Counter()
        for toks, _, _ in eventos_tokens:
            presentes = sorted(list(set(t for t in toks if t in termos_selecionados)))
            n_p = len(presentes)
            for i in range(n_p):
                for j in range(i + 1, n_p):
                    coocorrencias[(presentes[i], presentes[j])] += 1

        G = nx.Graph()

        for t in termos_selecionados:
            rotulo = self._formatar_rotulo_palavra(t)
            freq = palavra_stats[t]["freq"]
            divs = palavra_stats[t]["divergencias"]
            div_med = float(np.mean(divs)) if divs else 0.0
            G.add_node(
                t,
                label=rotulo,
                Label=rotulo,
                rotulo=rotulo,
                tipo_no="palavra",
                eixo=palavra_stats[t]["eixo"],
                frequencia=int(freq),
                divergencia_media=round(div_med, 4)
            )

        for (t1, t2), peso in coocorrencias.items():
            if peso >= min_coocorrencia:
                dist = round(1.0 / (1.0 + np.log1p(peso)), 4)
                G.add_edge(
                    t1,
                    t2,
                    weight=int(peso),
                    peso=int(peso),
                    distancia_semantica=dist
                )

        isolados = list(nx.isolates(G))
        G.remove_nodes_from(isolados)

        comunidades = nx_comm.louvain_communities(G, weight="weight", seed=42)
        mod_q = nx_comm.modularity(G, comunidades, weight="weight")

        deg = dict(G.degree(weight="weight"))
        bc = nx.betweenness_centrality(G, weight="distancia_semantica")

        pos_2d = nx.spring_layout(G, weight="weight", k=0.35, iterations=150, seed=42)

        for no in G.nodes:
            eixo = G.nodes[no].get("eixo", "politico")
            info_eixo = self.cores_eixos.get(eixo, self.cores_eixos["politico"])
            grau_val = deg.get(no, 1)
            bc_val = bc.get(no, 0.0)
            freq_val = G.nodes[no].get("frequencia", 10)
            tam_viz = float(np.clip(12.0 + 3.5 * np.log1p(freq_val) + 15.0 * bc_val, 12.0, 55.0))
            px, py = pos_2d[no]

            G.nodes[no]["modularity_class"] = info_eixo["class"]
            G.nodes[no]["comunidade_nome"] = info_eixo["nome"]
            G.nodes[no]["cor_hex"] = info_eixo["hex"]
            G.nodes[no]["betweenness"] = round(bc_val, 5)
            G.nodes[no]["grau_ponderado"] = grau_val
            G.nodes[no]["viz"] = {
                "size": round(tam_viz, 2),
                "color": {"r": info_eixo["r"], "g": info_eixo["g"], "b": info_eixo["b"], "a": 1.0},
                "position": {"x": round(float(px) * 1200.0, 2), "y": round(float(py) * 1200.0, 2), "z": 0.0}
            }

        analise = {
            "total_palavras": G.number_of_nodes(),
            "total_coocorrencias": G.number_of_edges(),
            "comunidades_louvain": len(comunidades),
            "modularidade_q": round(mod_q, 4),
            "top_termos_betweenness": [
                {"termo": G.nodes[no]["Label"], "score": round(score, 5), "comunidade": G.nodes[no].get("comunidade_nome")}
                for no, score in sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
            ]
        }
        self.logger.info(f"Grafo Semântico construído: {G.number_of_nodes()} palavras e {G.number_of_edges()} arestas.")
        return G, analise

    def exportar_gexf(self, G: nx.Graph, nome_arquivo: str = "grafo_semantico_palavras.gexf") -> str:
        caminho = self.dir_grafos / nome_arquivo
        nx.write_gexf(G, str(caminho))
        self.logger.info(f"Grafo semântico exportado para Gephi em: {caminho}")
        return str(caminho)

    def executar(self) -> Dict[str, Any]:
        G, analise = self.construir_grafo()
        caminho_gexf = self.exportar_gexf(G, "grafo_semantico_palavras.gexf")

        self.banco.salvar_resultado_estatistico(
            pergunta="grafo_semantico",
            chave="analise_nuvem_palavras_gephi",
            valor=analise.get("modularidade_q", 0.0),
            detalhes=analise
        )

        return {
            "grafo": G,
            "caminho_gexf": caminho_gexf,
            "analise": analise
        }

    def _adicionar_no_semantico(self, G: nx.Graph, row: Dict[str, str]) -> None:
        nid = row["id"]
        lbl = row["label"]
        eixo = row.get("eixo_tematico", "")
        freq = int(row.get("frequencia", 1))
        pr = float(row.get("pagerank", 0.0))
        bet = float(row.get("betweenness", 0.0))
        mod_class = int(row.get("modularity_class", 0))
        cor_hex = row.get("cor_hex", "#8E44AD")

        tam_s = float(round(float(6.0 + 35.0 * np.sqrt(pr * 50.0)), 2))
        G.add_node(
            nid,
            label=str(lbl),
            eixo_tematico=str(eixo),
            frequencia=int(freq),
            pagerank=float(pr),
            betweenness=float(bet),
            modularity_class=int(mod_class),
            cor_hex=str(cor_hex),
            size=tam_s
        )

    def executar_a_partir_de_pasta(self, dir_dados: str) -> Dict[str, Any]:
        p_dados = Path(dir_dados) if dir_dados else Path("resultados/grafos/arquivos")
        if not p_dados.exists():
            raise FileNotFoundError(f"Diretório informado não existe: {dir_dados}")
        p_zip = p_dados / "dados_grafos_gephi.zip"
        p_nos = p_dados / "02_grafo_semantico_palavras_nos.csv"
        p_arestas = p_dados / "02_grafo_semantico_palavras_arestas_coocorrencia.csv"

        self.logger.info(f"Construindo grafo Semântico a partir do pacote em {dir_dados}...")
        G = nx.Graph()

        if p_zip.exists():
            with zipfile.ZipFile(p_zip, "r") as zf:
                if "02_grafo_semantico_palavras_nos.csv" not in zf.namelist() or "02_grafo_semantico_palavras_arestas_coocorrencia.csv" not in zf.namelist():
                    raise FileNotFoundError(f"Arquivos do Grafo Semântico não encontrados dentro de {p_zip}")
                with zf.open("02_grafo_semantico_palavras_nos.csv") as f_nos:
                    reader = csv.DictReader(io.TextIOWrapper(f_nos, encoding="utf-8"))
                    for row in reader:
                        self._adicionar_no_semantico(G, row)
                with zf.open("02_grafo_semantico_palavras_arestas_coocorrencia.csv") as f_ar:
                    reader = csv.DictReader(io.TextIOWrapper(f_ar, encoding="utf-8"))
                    for row in reader:
                        G.add_edge(row["source"], row["target"], peso=float(row.get("peso", 1.0)), distancia_semantica=float(row.get("distancia_semantica", 1.0)), weight=float(row.get("weight", 1.0)))
        elif p_nos.exists() and p_arestas.exists():
            with open(p_nos, "r", encoding="utf-8") as f_nos:
                reader = csv.DictReader(f_nos)
                for row in reader:
                    self._adicionar_no_semantico(G, row)
            with open(p_arestas, "r", encoding="utf-8") as f_ar:
                reader = csv.DictReader(f_ar)
                for row in reader:
                    G.add_edge(row["source"], row["target"], peso=float(row.get("peso", 1.0)), distancia_semantica=float(row.get("distancia_semantica", 1.0)), weight=float(row.get("weight", 1.0)))
        else:
            raise FileNotFoundError(f"Arquivos do Grafo Semântico (CSVs ou dados_grafos_gephi.zip) não encontrados em {dir_dados}")

        caminho_gexf = self.exportar_gexf(G, "grafo_semantico_palavras.gexf")
        return {
            "grafo": G,
            "caminho_gexf": caminho_gexf,
            "total_nos": G.number_of_nodes(),
            "total_arestas": G.number_of_edges()
        }

    executar_a_partir_do_zenodo = executar_a_partir_de_pasta
