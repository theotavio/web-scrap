import gc
from pathlib import Path
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Set, Tuple
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

        self.veiculos_anonimizados = {
            1: ("Tradicional 1", "tradicional"),
            2: ("Tradicional 2", "tradicional"),
            3: ("Tradicional 3", "tradicional"),
            4: ("Tradicional 4", "tradicional"),
            5: ("Tradicional 5", "tradicional"),
            6: ("Digital 1", "digital_nativo"),
            7: ("Digital 2", "digital_nativo"),
            8: ("Digital 3", "digital_nativo"),
            9: ("Digital 4", "digital_nativo"),
            10: ("Digital 5", "digital_nativo")
        }

        self.cores_eixos = {
            "politico": {"r": 41, "g": 128, "b": 185},
            "economico": {"r": 243, "g": 156, "b": 18},
            "ambiental": {"r": 39, "g": 174, "b": 96},
            "social": {"r": 142, "g": 68, "b": 173},
            "internacional": {"r": 192, "g": 57, "b": 43},
            "esportivo": {"r": 230, "g": 126, "b": 34},
            "editorial": {"r": 44, "g": 62, "b": 80}
        }

        self.stopwords_gerais = {
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
            "poder360", "antagonista", "brasil247", "nexo", "g1", "uol", "após", "além", "sendo", "terá",
            "será", "está", "estão", "havia", "desde", "pouco", "muitos", "muitas", "feira", "desta",
            "deste", "segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo", "semana",
            "mês", "meses", "aponta", "mostra", "passa", "chega", "segue", "volta", "começa", "afirma",
            "pede", "vê", "revela", "traz", "diz", "dizer", "levar", "sair", "dar", "fica", "ter", "vir",
            "ver", "vai", "vão", "ia", "virou", "ficou", "deu", "foi", "era", "são", "tem", "entre",
            "planalto", "assim", "cada", "tanto", "quanto", "sabe", "tinha", "vamos", "então", "aqui"
        }

    def limpar_grafos_antigos(self) -> None:
        arquivos_antigos = [
            "rede_bipartida.gexf",
            "rede_veiculos.gexf",
            "rede_macro_eventos.gexf"
        ]
        for arq in arquivos_antigos:
            p = self.dir_relatorios / arq
            if p.exists():
                p.unlink()
                self.logger.info(f"Arquivo antigo removido: {p}")

    def obter_entidades_pessoas_para_filtro(self) -> Set[str]:
        con = self.banco.obter_conexao()
        cur = con.cursor()
        cur.execute("""
            SELECT LOWER(texto_entidade)
            FROM entidades_mencoes
            WHERE tipo_entidade = 'PER'
            GROUP BY LOWER(texto_entidade)
            HAVING COUNT(*) >= 50
        """)
        nomes_raw = [r[0] for r in cur.fetchall()]
        con.close()

        tokens_nomes = set()
        for n in nomes_raw:
            for t in re.findall(r"\b[a-záéíóúãõâêîôûç]{3,}\b", n):
                tokens_nomes.add(t)
        return tokens_nomes

    def construir_grafo_semantico(
        self,
        palavras_por_eixo: int = 250,
        max_coocorrencias_por_palavra: int = 2,
        max_palavras_por_veiculo: int = 15
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        self.logger.info("Iniciando mineracao de alta definicao para o Grafo Semantico...")
        tokens_nomes = self.obter_entidades_pessoas_para_filtro()
        stopwords_totais = self.stopwords_gerais | tokens_nomes

        eixos = ["politico", "economico", "ambiental", "social", "internacional", "esportivo"]
        con = self.banco.obter_conexao()
        cur = con.cursor()

        palavra_info: Dict[str, Dict[str, Any]] = {}

        for eixo in eixos:
            cur.execute(f"""
                SELECT titulo, veiculo_id, sentimento_polaridade, score_clickbait
                FROM materias
                WHERE eixo_tematico = '{eixo}' AND id % 3 = 0
                LIMIT 35000
            """)
            rows = cur.fetchall()
            cnt = Counter()
            for r in rows:
                tokens = re.findall(r"\b[a-záéíóúãõâêîôûç]{4,}\b", r[0].lower())
                filtrados = [t for t in tokens if t not in stopwords_totais and len(t) >= 4]
                cnt.update(filtrados)

            for w, freq in cnt.most_common(palavras_por_eixo):
                if w not in palavra_info:
                    palavra_info[w] = {
                        "eixo": eixo,
                        "frequencia": freq,
                        "sentimentos": [],
                        "clickbaits": [],
                        "veiculos": defaultdict(int)
                    }

        todas_palavras_set = set(palavra_info.keys())
        coocorrencias: Counter = Counter()

        for eixo in eixos:
            cur.execute(f"""
                SELECT titulo, veiculo_id, sentimento_polaridade, score_clickbait
                FROM materias
                WHERE eixo_tematico = '{eixo}' AND id % 3 = 0
                LIMIT 35000
            """)
            rows = cur.fetchall()
            for r in rows:
                v_id = r[1]
                sent = float(r[2] or 0.0)
                cb = float(r[3] or 0.0)
                tokens = set(re.findall(r"\b[a-záéíóúãõâêîôûç]{4,}\b", r[0].lower()))
                presentes = [t for t in tokens if t in todas_palavras_set]

                for w in presentes:
                    palavra_info[w]["sentimentos"].append(sent)
                    palavra_info[w]["clickbaits"].append(cb)
                    palavra_info[w]["veiculos"][v_id] += 1

                for i in range(len(presentes)):
                    for j in range(i + 1, len(presentes)):
                        w1, w2 = sorted([presentes[i], presentes[j]])
                        coocorrencias[(w1, w2)] += 1

        con.close()

        cooc_por_no: Dict[str, List[Tuple[str, float]]] = {}
        for (w1, w2), count in coocorrencias.items():
            if count >= 3:
                cooc_por_no.setdefault(w1, []).append((w2, float(count)))
                cooc_por_no.setdefault(w2, []).append((w1, float(count)))

        arestas_podadas: Set[Tuple[str, str, float, str]] = set()

        for u, vizinhos in cooc_por_no.items():
            vizinhos_ord = sorted(vizinhos, key=lambda x: x[1], reverse=True)[:max_coocorrencias_por_palavra]
            for v, w in vizinhos_ord:
                edge_key = tuple(sorted([f"palavra_{u}", f"palavra_{v}"]))
                arestas_podadas.add((edge_key[0], edge_key[1], w, "coocorrencia"))

        for v_id in range(1, 11):
            no_veic = f"veiculo_{v_id}"
            pals_veic = []
            for w, info in palavra_info.items():
                qtd = info["veiculos"].get(v_id, 0)
                if qtd >= 2:
                    pals_veic.append((f"palavra_{w}", float(qtd)))
            pals_ord = sorted(pals_veic, key=lambda x: x[1], reverse=True)[:max_palavras_por_veiculo]
            for pal_node, w in pals_ord:
                edge_key = tuple(sorted([no_veic, pal_node]))
                arestas_podadas.add((edge_key[0], edge_key[1], w, "enquadramento"))

        G = nx.Graph()

        max_freq = max(info["frequencia"] for info in palavra_info.values()) if palavra_info else 1

        for v_id, (v_label, v_tipo) in self.veiculos_anonimizados.items():
            cor = self.cores_eixos["editorial"]
            G.add_node(
                f"veiculo_{v_id}",
                Label=v_label,
                rotulo=v_label,
                tipo_no="veiculo",
                tipo_editorial=v_tipo,
                eixo="editorial",
                frequencia=10000,
                sentimento_medio=0.0,
                clickbait_medio=0.0,
                size=40.0,
                viz={"size": 40.0, "color": cor}
            )

        for w, info in palavra_info.items():
            sents = info["sentimentos"]
            cbs = info["clickbaits"]
            s_med = float(np.mean(sents)) if sents else 0.0
            cb_med = float(np.mean(cbs)) if cbs else 0.0
            freq = float(info["frequencia"])
            tam = 8.0 + 18.0 * (np.log1p(freq) / np.log1p(max_freq))
            cor = self.cores_eixos.get(info["eixo"], {"r": 120, "g": 120, "b": 120})

            G.add_node(
                f"palavra_{w}",
                Label=w.capitalize(),
                rotulo=w.capitalize(),
                tipo_no="palavra",
                eixo=info["eixo"],
                frequencia=int(info["frequencia"]),
                sentimento_medio=round(s_med, 4),
                clickbait_medio=round(cb_med, 4),
                size=round(float(tam), 2),
                viz={"size": round(float(tam), 2), "color": cor}
            )

        max_w = max(w for u, v, w, t in arestas_podadas) if arestas_podadas else 1.0
        min_w = min(w for u, v, w, t in arestas_podadas) if arestas_podadas else 0.0

        for u, v, w, t in arestas_podadas:
            w_norm = 0.3 + 2.2 * ((w - min_w) / (max_w - min_w + 1e-6))
            G.add_edge(
                u,
                v,
                weight=round(float(w_norm), 3),
                peso_bruto=int(w),
                tipo_aresta=t,
                distancia_semantica=round(1.0 / (1.0 + np.log1p(w)), 4)
            )

        pos = nx.spring_layout(G, k=0.18, iterations=80, seed=42)
        for n in G.nodes():
            x, y = pos[n]
            G.nodes[n]["viz"]["position"] = {
                "x": round(float(x * 1200.0), 2),
                "y": round(float(y * 1200.0), 2),
                "z": 0.0
            }

        del coocorrencias
        del palavra_info
        gc.collect()

        metricas_algoritmos = self._calcular_metricas_grafo(G)
        return G, metricas_algoritmos

    def _calcular_metricas_grafo(self, G: nx.Graph) -> Dict[str, Any]:
        self.logger.info("Calculando metricas analiticas de algoritmos no Grafo Semantico...")

        comunidades = nx_comm.louvain_communities(G, weight="weight", resolution=1.0)
        mod_q = nx_comm.modularity(G, comunidades, weight="weight")

        for idx_c, comm in enumerate(comunidades):
            for no in comm:
                G.nodes[no]["modularity_class"] = idx_c

        veiculos_trad = [f"veiculo_{i}" for i in range(1, 6)]
        veiculos_dig = [f"veiculo_{i}" for i in range(6, 11)]

        dists_trad_trad = []
        dists_dig_dig = []
        dists_cruzadas = []
        caminhos_exemplares = []

        for i in range(1, 6):
            for j in range(i + 1, 6):
                v1, v2 = f"veiculo_{i}", f"veiculo_{j}"
                if nx.has_path(G, v1, v2):
                    d = nx.shortest_path_length(G, v1, v2, weight="distancia_semantica")
                    dists_trad_trad.append(d)

        for i in range(6, 11):
            for j in range(i + 1, 11):
                v1, v2 = f"veiculo_{i}", f"veiculo_{j}"
                if nx.has_path(G, v1, v2):
                    d = nx.shortest_path_length(G, v1, v2, weight="distancia_semantica")
                    dists_dig_dig.append(d)

        for i in range(1, 6):
            for j in range(6, 11):
                v1, v2 = f"veiculo_{i}", f"veiculo_{j}"
                if nx.has_path(G, v1, v2):
                    d = nx.shortest_path_length(G, v1, v2, weight="distancia_semantica")
                    caminho = nx.shortest_path(G, v1, v2, weight="distancia_semantica")
                    dists_cruzadas.append(d)
                    if len(caminho) >= 3 and len(caminhos_exemplares) < 5:
                        caminhos_exemplares.append({
                            "origem": G.nodes[v1].get("Label"),
                            "destino": G.nodes[v2].get("Label"),
                            "distancia": round(float(d), 4),
                            "termo_ponte": G.nodes[caminho[1]].get("Label")
                        })

        bc = nx.betweenness_centrality(G, weight="distancia_semantica", k=min(200, G.number_of_nodes()))
        termos_bc = [(n, val) for n, val in bc.items() if G.nodes[n].get("tipo_no") == "palavra"]
        top_pontes = [
            {"termo": G.nodes[no].get("Label"), "eixo": G.nodes[no].get("eixo"), "score": round(float(score), 6)}
            for no, score in sorted(termos_bc, key=lambda x: x[1], reverse=True)[:10]
        ]

        bfs_saltos = []
        for v in veiculos_trad + veiculos_dig:
            lengths = nx.single_source_shortest_path_length(G, v)
            for v_alvo in veiculos_trad + veiculos_dig:
                if v != v_alvo and v_alvo in lengths:
                    bfs_saltos.append(lengths[v_alvo])

        resultado = {
            "total_nos": G.number_of_nodes(),
            "total_arestas": G.number_of_edges(),
            "total_comunidades_louvain": len(comunidades),
            "modularidade_q": round(float(mod_q), 4),
            "distancia_media_dijkstra_tradicional": float(np.mean(dists_trad_trad)) if dists_trad_trad else 0.0,
            "distancia_media_dijkstra_digital": float(np.mean(dists_dig_dig)) if dists_dig_dig else 0.0,
            "distancia_media_dijkstra_cruzada": float(np.mean(dists_cruzadas)) if dists_cruzadas else 0.0,
            "caminhos_geodesicos_exemplares": caminhos_exemplares,
            "top_termos_ponte_betweenness": top_pontes,
            "diametro_bfs_entre_veiculos": max(bfs_saltos) if bfs_saltos else 0,
            "media_saltos_bfs": float(np.mean(bfs_saltos)) if bfs_saltos else 0.0
        }
        return resultado

    def exportar_gexf(self, G: nx.Graph, nome_arquivo: str) -> str:
        caminho = self.dir_relatorios / nome_arquivo
        nx.write_gexf(G, str(caminho))
        self.logger.info(f"Grafo exportado para Gephi em: {caminho}")
        return str(caminho)

    def executar(self) -> Dict[str, Any]:
        self.limpar_grafos_antigos()
        G_semantico, analise = self.construir_grafo_semantico()
        caminho_grafo = self.exportar_gexf(G_semantico, "grafo_semantico_palavras.gexf")

        self.banco.salvar_resultado_estatistico(
            pergunta="grafo_semantico",
            chave="analise_rede_nuvem_palavras",
            valor=analise.get("modularidade_q", 0.0),
            detalhes=analise
        )

        metricas = {
            "grafo_nos": G_semantico.number_of_nodes(),
            "grafo_arestas": G_semantico.number_of_edges(),
            "caminho_gexf": caminho_grafo,
            "analise_semantica": analise,
            "grafo": G_semantico
        }

        self.logger.info(f"Grafo Semantico gerado com sucesso: {analise['total_nos']} nos e {analise['total_arestas']} arestas.")
        return metricas
