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

        self.veiculos_pos = {
            "Tradicional 1": (-360.0, 160.0),
            "Tradicional 2": (-400.0, 80.0),
            "Tradicional 3": (-380.0, 0.0),
            "Tradicional 4": (-400.0, -80.0),
            "Tradicional 5": (-360.0, -160.0),
            "Digital 1": (360.0, 160.0),
            "Digital 2": (400.0, 80.0),
            "Digital 3": (380.0, 0.0),
            "Digital 4": (400.0, -80.0),
            "Digital 5": (360.0, -160.0),
        }

        self.cores_eixos = {
            "politico": {"r": 41, "g": 128, "b": 185},
            "economico": {"r": 243, "g": 156, "b": 18},
            "ambiental": {"r": 39, "g": 174, "b": 96},
            "social": {"r": 142, "g": 68, "b": 173},
            "internacional": {"r": 192, "g": 57, "b": 43},
            "esportivo": {"r": 230, "g": 126, "b": 34},
        }
        self.cor_tradicional = {"r": 27, "g": 79, "b": 114}
        self.cor_digital = {"r": 211, "g": 84, "b": 0}

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
            "planalto", "assim", "cada", "tanto", "quanto", "sabe", "tinha", "vamos", "então", "aqui",
            "deve", "fazer", "quer", "vai", "vir", "dar", "ter", "estar", "dizer", "poder", "ver", "fica",
            "todo", "toda", "todos", "todas", "sido", "terem", "dizer", "acordo", "passado", "chegou"
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
        palavras_por_eixo: int = 25,
        amostras_por_veiculo_eixo: int = 1500
    ) -> Tuple[nx.Graph, Dict[str, Any]]:
        self.logger.info("Iniciando mineracao balanceada e simetrica para o Grafo Semantico...")
        tokens_nomes = self.obter_entidades_pessoas_para_filtro()
        stopwords_totais = self.stopwords_gerais | tokens_nomes

        eixos = ["politico", "economico", "ambiental", "social", "internacional", "esportivo"]
        con = self.banco.obter_conexao()
        cur = con.cursor()

        palavra_info: Dict[str, Dict[str, Any]] = {}

        for eixo in eixos:
            cnt = Counter()
            for v in range(1, 11):
                cur.execute(f"""
                    SELECT titulo FROM materias
                    WHERE veiculo_id = {v} AND eixo_tematico = '{eixo}'
                    LIMIT {amostras_por_veiculo_eixo}
                """)
                rows = cur.fetchall()
                for r in rows:
                    tokens = re.findall(r"\b[a-záéíóúãõâêîôûç]{4,}\b", r[0].lower())
                    filtrados = [t for t in tokens if t not in stopwords_totais and len(t) >= 4]
                    cnt.update(filtrados)

            for w, freq in cnt.most_common(palavras_por_eixo):
                termo = w.capitalize()
                if termo not in palavra_info:
                    palavra_info[termo] = {
                        "eixo": eixo,
                        "frequencia": freq,
                        "tradicional_count": 0,
                        "digital_count": 0,
                        "veiculos": defaultdict(int)
                    }

        todas_palavras_set = {k.lower(): k for k in palavra_info.keys()}
        coocorrencias: Counter = Counter()

        for eixo in eixos:
            for v in range(1, 11):
                cur.execute(f"""
                    SELECT titulo FROM materias
                    WHERE veiculo_id = {v} AND eixo_tematico = '{eixo}'
                    LIMIT {amostras_por_veiculo_eixo}
                """)
                rows = cur.fetchall()
                for r in rows:
                    tokens = set(re.findall(r"\b[a-záéíóúãõâêîôûç]{4,}\b", r[0].lower()))
                    presentes = [todas_palavras_set[t] for t in tokens if t in todas_palavras_set]
                    for p in presentes:
                        palavra_info[p]["veiculos"][v] += 1
                        if v <= 5:
                            palavra_info[p]["tradicional_count"] += 1
                        else:
                            palavra_info[p]["digital_count"] += 1

                    for i in range(len(presentes)):
                        for j in range(i + 1, len(presentes)):
                            p1, p2 = sorted([presentes[i], presentes[j]])
                            coocorrencias[(p1, p2)] += 1

        con.close()

        G = nx.Graph()

        for v_id in range(1, 6):
            nome = f"Tradicional {v_id}"
            px, py = self.veiculos_pos[nome]
            G.add_node(
                nome,
                Label=nome,
                rotulo=nome,
                tipo_no="veiculo",
                grupo_editorial="Tradicional",
                eixo="Imprensa Tradicional",
                size=55.0,
                viz={"size": 55.0, "color": self.cor_tradicional, "position": {"x": px, "y": py, "z": 0.0}}
            )

        for v_id in range(1, 6):
            nome = f"Digital {v_id}"
            px, py = self.veiculos_pos[nome]
            G.add_node(
                nome,
                Label=nome,
                rotulo=nome,
                tipo_no="veiculo",
                grupo_editorial="Digital Nativo",
                eixo="Nativos Digitais",
                size=55.0,
                viz={"size": 55.0, "color": self.cor_digital, "position": {"x": px, "y": py, "z": 0.0}}
            )

        y_offsets = {
            "politico": 180.0,
            "economico": 80.0,
            "internacional": -10.0,
            "ambiental": -90.0,
            "social": -170.0,
            "esportivo": -250.0
        }

        np.random.seed(42)
        max_freq = max(info["frequencia"] for info in palavra_info.values()) if palavra_info else 1

        for termo, info in palavra_info.items():
            t_cnt = info["tradicional_count"]
            d_cnt = info["digital_count"]
            total_cnt = t_cnt + d_cnt + 1e-5
            polarizacao = (d_cnt - t_cnt) / total_cnt

            if polarizacao < -0.15:
                enq = "Viés Tradicional"
                cor = self.cor_tradicional
            elif polarizacao > 0.15:
                enq = "Viés Digital"
                cor = self.cor_digital
            else:
                enq = "Consenso Central"
                cor = self.cores_eixos.get(info["eixo"], {"r": 100, "g": 100, "b": 100})

            base_x = polarizacao * 250.0 + np.random.uniform(-25, 25)
            base_y = y_offsets.get(info["eixo"], 0.0) + np.random.uniform(-20, 20)

            freq = info["frequencia"]
            tam = 22.0 + 20.0 * (np.log1p(freq) / np.log1p(max_freq))

            G.add_node(
                termo,
                Label=termo,
                rotulo=termo,
                tipo_no="palavra",
                eixo=info["eixo"],
                enquadramento=enq,
                polarizacao_editorial=round(float(polarizacao), 4),
                frequencia=int(freq),
                size=round(float(tam), 2),
                viz={"size": round(float(tam), 2), "color": cor, "position": {"x": round(float(base_x), 2), "y": round(float(base_y), 2), "z": 0.0}}
            )

        cooc_por_no = defaultdict(list)
        for (p1, p2), count in coocorrencias.items():
            if count >= 3:
                cooc_por_no[p1].append((p2, count))
                cooc_por_no[p2].append((p1, count))

        arestas_adicionadas = set()
        for p1, vizinhos in cooc_por_no.items():
            viz_ord = sorted(vizinhos, key=lambda x: x[1], reverse=True)[:2]
            for p2, count in viz_ord:
                chave = tuple(sorted([p1, p2]))
                if chave not in arestas_adicionadas and p1 in G and p2 in G:
                    w_norm = 0.5 + 1.0 * min(1.0, count / 50.0)
                    G.add_edge(
                        p1,
                        p2,
                        weight=round(float(w_norm), 3),
                        tipo_aresta="coocorrencia",
                        distancia=round(1.0 / (1.0 + np.log1p(count)), 4),
                        viz={"color": {"r": 210, "g": 215, "b": 220, "a": 0.3}}
                    )
                    arestas_adicionadas.add(chave)

        for termo, info in palavra_info.items():
            for v_id, count in info["veiculos"].items():
                if count >= 5:
                    v_nome = f"Tradicional {v_id}" if v_id <= 5 else f"Digital {v_id - 5}"
                    chave = tuple(sorted([v_nome, termo]))
                    if chave not in arestas_adicionadas and v_nome in G and termo in G:
                        w_norm = 0.5 + 1.0 * min(1.0, count / 40.0)
                        cor_aresta = {"r": 180, "g": 200, "b": 220, "a": 0.35} if v_id <= 5 else {"r": 240, "g": 200, "b": 160, "a": 0.35}
                        G.add_edge(
                            v_nome,
                            termo,
                            weight=round(float(w_norm), 3),
                            tipo_aresta="enquadramento",
                            distancia=round(1.0 / (1.0 + np.log1p(count)), 4),
                            viz={"color": cor_aresta}
                        )
                        arestas_adicionadas.add(chave)

        del coocorrencias
        del palavra_info
        gc.collect()

        metricas = self._calcular_metricas_grafo(G)
        return G, metricas

    def _calcular_metricas_grafo(self, G: nx.Graph) -> Dict[str, Any]:
        self.logger.info("Calculando metricas analiticas para o Grafo Semantico Polarizado...")

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
                    caminho = nx.shortest_path(G, v1, v2, weight="distancia")
                    if len(caminho) >= 3 and len(caminhos_exemplares) < 5:
                        caminhos_exemplares.append({
                            "origem": v1,
                            "destino": v2,
                            "distancia": round(float(d), 4),
                            "termo_ponte": caminho[1]
                        })

        bc = nx.betweenness_centrality(G, weight="distancia")
        termos_bc = [(n, val) for n, val in bc.items() if G.nodes[n].get("tipo_no") == "palavra"]
        top_pontes = [
            {"termo": no, "eixo": G.nodes[no].get("eixo"), "score": round(float(score), 6)}
            for no, score in sorted(termos_bc, key=lambda x: x[1], reverse=True)[:10]
        ]

        bfs_saltos = []
        for v in veics_trad + veics_dig:
            lengths = nx.single_source_shortest_path_length(G, v)
            for v_alvo in veics_trad + veics_dig:
                if v != v_alvo and v_alvo in lengths:
                    bfs_saltos.append(lengths[v_alvo])

        resultado = {
            "total_nos": G.number_of_nodes(),
            "total_arestas": G.number_of_edges(),
            "total_comunidades_louvain": len(comunidades),
            "modularidade_q": round(float(mod_q), 4),
            "distancia_media_dijkstra_tradicional": float(np.mean(d_trad)) if d_trad else 0.0,
            "distancia_media_dijkstra_digital": float(np.mean(d_dig)) if d_dig else 0.0,
            "distancia_media_dijkstra_cruzada": float(np.mean(d_cross)) if d_cross else 0.0,
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
