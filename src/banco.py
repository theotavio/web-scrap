import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from src.configuracao import obter_configuracao
from src.registro import obter_logger


class BancoDados:
    def __init__(self, caminho_banco: Optional[str] = None):
        config = obter_configuracao()
        if caminho_banco is None:
            caminho_banco = config.obter("geral.banco_dados", "dados/jornalismo.db")
        self.caminho = Path(caminho_banco)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.logger = obter_logger("banco")
        self.inicializar()

    def obter_conexao(self) -> sqlite3.Connection:
        con = sqlite3.connect(str(self.caminho), timeout=60.0)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL;")
        con.execute("PRAGMA synchronous=NORMAL;")
        con.execute("PRAGMA foreign_keys=ON;")
        con.execute("PRAGMA cache_size=-64000;")
        con.execute("PRAGMA mmap_size=268435456;")
        con.execute("PRAGMA temp_store=MEMORY;")
        return con

    def inicializar(self) -> None:
        con = self.obter_conexao()
        cur = con.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS veiculos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                codigo TEXT UNIQUE NOT NULL,
                nome TEXT NOT NULL,
                dominio TEXT NOT NULL,
                tipo TEXT NOT NULL,
                ano_fundacao INTEGER
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS urls_coletadas (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                veiculo_id INTEGER NOT NULL,
                url TEXT UNIQUE NOT NULL,
                fonte_coleta TEXT NOT NULL,
                timestamp_cdx TEXT,
                data_prevista TEXT,
                eixo_tematico TEXT,
                status_extracao TEXT DEFAULT 'pendente',
                criado_em DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (veiculo_id) REFERENCES veiculos(id)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS materias (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url_id INTEGER UNIQUE,
                veiculo_id INTEGER NOT NULL,
                url TEXT UNIQUE NOT NULL,
                titulo TEXT NOT NULL,
                corpo TEXT NOT NULL,
                resumo TEXT,
                data_publicacao TEXT NOT NULL,
                ano INTEGER NOT NULL,
                mes INTEGER NOT NULL,
                ano_eleitoral INTEGER DEFAULT 0,
                autor TEXT,
                eixo_tematico TEXT,
                hash_conteudo TEXT UNIQUE NOT NULL,
                tamanho_texto INTEGER NOT NULL,
                score_clickbait REAL DEFAULT 0.0,
                sentimento_polaridade REAL DEFAULT 0.0,
                sentimento_pos REAL DEFAULT 0.0,
                sentimento_neg REAL DEFAULT 0.0,
                sentimento_neu REAL DEFAULT 0.0,
                evento_id INTEGER DEFAULT -1,
                criado_em DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (url_id) REFERENCES urls_coletadas(id),
                FOREIGN KEY (veiculo_id) REFERENCES veiculos(id)
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                materia_id INTEGER PRIMARY KEY,
                vetor BLOB NOT NULL,
                dimensao INTEGER NOT NULL,
                FOREIGN KEY (materia_id) REFERENCES materias(id) ON DELETE CASCADE
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS entidades_mencoes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                materia_id INTEGER NOT NULL,
                texto_entidade TEXT NOT NULL,
                tipo_entidade TEXT NOT NULL,
                contagem INTEGER DEFAULT 1,
                polaridade_atribuicao REAL DEFAULT 0.0,
                FOREIGN KEY (materia_id) REFERENCES materias(id) ON DELETE CASCADE
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS clusters_eventos (
                id INTEGER PRIMARY KEY,
                nome_evento TEXT,
                data_inicio TEXT,
                data_fim TEXT,
                total_materias INTEGER DEFAULT 0,
                total_veiculos INTEGER DEFAULT 0,
                eixo_predominante TEXT,
                divergencia_sentimento REAL DEFAULT 0.0
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS resultados_estatisticos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pergunta TEXT NOT NULL,
                chave_metrica TEXT NOT NULL,
                valor_numerico REAL,
                detalhes_json TEXT,
                calculado_em DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_urls_veiculo_status ON urls_coletadas(veiculo_id, status_extracao);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_urls_status_id ON urls_coletadas(status_extracao, id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_urls_veic_status_id ON urls_coletadas(veiculo_id, status_extracao, id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_urls_url ON urls_coletadas(url);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_materias_veiculo ON materias(veiculo_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_materias_data ON materias(data_publicacao);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_materias_ano ON materias(ano);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_materias_eleitoral ON materias(ano_eleitoral);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_materias_evento ON materias(evento_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_entidades_materia ON entidades_mencoes(materia_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_entidades_texto ON entidades_mencoes(texto_entidade);")

        con.commit()
        con.close()
        self._sincronizar_veiculos()

    def _sincronizar_veiculos(self) -> None:
        config = obter_configuracao()
        veiculos_cfg = config.obter("veiculos", {})
        con = self.obter_conexao()
        cur = con.cursor()
        for cod, info in veiculos_cfg.items():
            cur.execute("""
                INSERT INTO veiculos (codigo, nome, dominio, tipo, ano_fundacao)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(codigo) DO UPDATE SET
                    nome=excluded.nome,
                    dominio=excluded.dominio,
                    tipo=excluded.tipo,
                    ano_fundacao=excluded.ano_fundacao;
            """, (cod, info.get("nome"), info.get("dominio"), info.get("tipo"), info.get("ano_fundacao")))
        con.commit()
        con.close()

    def resetar(self) -> None:
        if self.caminho.exists():
            self.caminho.unlink()
        self.inicializar()

    def inserir_urls_em_lote(self, urls_dados: List[Tuple[int, str, str, Optional[str], Optional[str], Optional[str]]]) -> int:
        if not urls_dados:
            return 0
        con = self.obter_conexao()
        cur = con.cursor()
        cur.executemany("""
            INSERT OR IGNORE INTO urls_coletadas (
                veiculo_id, url, fonte_coleta, timestamp_cdx, data_prevista, eixo_tematico
            ) VALUES (?, ?, ?, ?, ?, ?);
        """, urls_dados)
        inseridos = cur.rowcount
        con.commit()
        con.close()
        return inseridos

    def contar_urls_pendentes(self, veiculo_id: Optional[int] = None, status: str = "pendente") -> int:
        con = self.obter_conexao()
        cur = con.cursor()
        if veiculo_id:
            cur.execute("SELECT COUNT(*) FROM urls_coletadas WHERE status_extracao = ? AND veiculo_id = ?", (status, veiculo_id))
        else:
            cur.execute("SELECT COUNT(*) FROM urls_coletadas WHERE status_extracao = ?", (status,))
        total = cur.fetchone()[0]
        con.close()
        return int(total)

    def obter_urls_pendentes(self, limite: Optional[int] = None, veiculo_id: Optional[int] = None, status: str = "pendente") -> List[Dict[str, Any]]:
        con = self.obter_conexao()
        cur = con.cursor()
        query = """
            SELECT u.id, u.veiculo_id, u.url, u.fonte_coleta, u.timestamp_cdx, u.data_prevista, u.eixo_tematico,
                   v.codigo as veiculo_codigo, v.dominio as veiculo_dominio
            FROM urls_coletadas u
            JOIN veiculos v ON u.veiculo_id = v.id
            WHERE u.status_extracao = ?
        """
        params = [status]
        if veiculo_id:
            query += " AND u.veiculo_id = ?"
            params.append(veiculo_id)
        query += " ORDER BY u.id ASC"
        if limite:
            query += f" LIMIT {limite}"
        cur.execute(query, params)
        linhas = [dict(row) for row in cur.fetchall()]
        con.close()
        return linhas

    def resetar_status_urls(self, status_origem: str = "erro_download", status_destino: str = "pendente", veiculo_id: Optional[int] = None) -> int:
        con = self.obter_conexao()
        cur = con.cursor()
        if veiculo_id:
            cur.execute("UPDATE urls_coletadas SET status_extracao = ? WHERE status_extracao = ? AND veiculo_id = ?", (status_destino, status_origem, veiculo_id))
        else:
            cur.execute("UPDATE urls_coletadas SET status_extracao = ? WHERE status_extracao = ?", (status_destino, status_origem))
        afetados = cur.rowcount
        con.commit()
        con.close()
        return afetados

    def atualizar_status_url(self, url_id: int, status: str) -> None:
        con = self.obter_conexao()
        cur = con.cursor()
        cur.execute("UPDATE urls_coletadas SET status_extracao = ? WHERE id = ?", (status, url_id))
        con.commit()
        con.close()

    def inserir_materia(self, materia: Dict[str, Any]) -> Optional[int]:
        con = self.obter_conexao()
        cur = con.cursor()
        try:
            cur.execute("""
                INSERT INTO materias (
                    url_id, veiculo_id, url, titulo, corpo, resumo,
                    data_publicacao, ano, mes, ano_eleitoral, autor,
                    eixo_tematico, hash_conteudo, tamanho_texto
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                materia.get("url_id"),
                materia["veiculo_id"],
                materia["url"],
                materia["titulo"],
                materia["corpo"],
                materia.get("resumo"),
                materia["data_publicacao"],
                materia["ano"],
                materia["mes"],
                materia.get("ano_eleitoral", 0),
                materia.get("autor"),
                materia.get("eixo_tematico"),
                materia["hash_conteudo"],
                materia["tamanho_texto"]
            ))
            materia_id = cur.lastrowid
            if materia.get("url_id"):
                cur.execute("UPDATE urls_coletadas SET status_extracao = 'extraido' WHERE id = ?", (materia["url_id"],))
            con.commit()
            return materia_id
        except sqlite3.IntegrityError:
            if materia.get("url_id"):
                cur.execute("UPDATE urls_coletadas SET status_extracao = 'duplicado' WHERE id = ?", (materia["url_id"],))
                con.commit()
            return None
        finally:
            con.close()

    def limpar_materias_invalidas_ou_curtas(self, min_tamanho: int = 150) -> int:
        con = self.obter_conexao()
        cur = con.cursor()
        cur.execute("DELETE FROM materias WHERE tamanho_texto < ? OR corpo IS NULL OR titulo IS NULL", (min_tamanho,))
        removidos = cur.rowcount
        con.commit()
        con.close()
        return removidos

    def salvar_embeddings_em_lote(self, embeddings_dados: List[Tuple[int, bytes, int]]) -> None:
        if not embeddings_dados:
            return
        con = self.obter_conexao()
        cur = con.cursor()
        cur.executemany("""
            INSERT INTO embeddings (materia_id, vetor, dimensao)
            VALUES (?, ?, ?)
            ON CONFLICT(materia_id) DO UPDATE SET
                vetor=excluded.vetor,
                dimensao=excluded.dimensao;
        """, embeddings_dados)
        con.commit()
        con.close()

    def salvar_features_materia(self, materia_id: int, clickbait: float, pol: float, pos: float, neg: float, neu: float) -> None:
        con = self.obter_conexao()
        cur = con.cursor()
        cur.execute("""
            UPDATE materias
            SET score_clickbait = ?,
                sentimento_polaridade = ?,
                sentimento_pos = ?,
                sentimento_neg = ?,
                sentimento_neu = ?
            WHERE id = ?
        """, (clickbait, pol, pos, neg, neu, materia_id))
        con.commit()
        con.close()

    def salvar_features_em_lote(self, lista_features: List[Tuple[float, float, float, float, float, int]]) -> None:
        if not lista_features:
            return
        con = self.obter_conexao()
        cur = con.cursor()
        cur.executemany("""
            UPDATE materias
            SET score_clickbait = ?,
                sentimento_polaridade = ?,
                sentimento_pos = ?,
                sentimento_neg = ?,
                sentimento_neu = ?
            WHERE id = ?
        """, lista_features)
        con.commit()
        con.close()

    def salvar_entidades_mencoes(self, materia_id: int, entidades: List[Tuple[str, str, int, float]]) -> None:
        if not entidades:
            return
        con = self.obter_conexao()
        cur = con.cursor()
        cur.execute("DELETE FROM entidades_mencoes WHERE materia_id = ?", (materia_id,))
        cur.executemany("""
            INSERT INTO entidades_mencoes (materia_id, texto_entidade, tipo_entidade, contagem, polaridade_atribuicao)
            VALUES (?, ?, ?, ?, ?)
        """, [(materia_id, e[0], e[1], e[2], e[3]) for e in entidades])
        con.commit()
        con.close()

    def salvar_entidades_em_lote(self, lista_entidades_materia: List[Tuple[int, List[Tuple[str, str, int, float]]]]) -> None:
        if not lista_entidades_materia:
            return
        con = self.obter_conexao()
        cur = con.cursor()
        ids = [item[0] for item in lista_entidades_materia]
        if ids:
            placeholders = ",".join("?" * len(ids))
            cur.execute(f"DELETE FROM entidades_mencoes WHERE materia_id IN ({placeholders})", ids)

        todas_linhas = []
        for mat_id, ents in lista_entidades_materia:
            for e in ents:
                todas_linhas.append((mat_id, e[0], e[1], e[2], e[3]))

        if todas_linhas:
            cur.executemany("""
                INSERT INTO entidades_mencoes (materia_id, texto_entidade, tipo_entidade, contagem, polaridade_atribuicao)
                VALUES (?, ?, ?, ?, ?)
            """, todas_linhas)
        con.commit()
        con.close()

    def atualizar_cluster_materia(self, materia_id: int, evento_id: int) -> None:
        con = self.obter_conexao()
        cur = con.cursor()
        cur.execute("UPDATE materias SET evento_id = ? WHERE id = ?", (evento_id, materia_id))
        con.commit()
        con.close()

    def atualizar_clusters_em_lote(self, clusters_materias: List[Tuple[int, int]]) -> None:
        if not clusters_materias:
            return
        con = self.obter_conexao()
        cur = con.cursor()
        cur.executemany("UPDATE materias SET evento_id = ? WHERE id = ?", [(c[1], c[0]) for c in clusters_materias])
        con.commit()
        con.close()

    def atualizar_eixos_tematicos_em_lote(self, eixos_materias: List[Tuple[str, int]]) -> None:
        if not eixos_materias:
            return
        con = self.obter_conexao()
        cur = con.cursor()
        cur.executemany("UPDATE materias SET eixo_tematico = ? WHERE id = ?", eixos_materias)
        con.commit()
        con.close()

    def salvar_resumo_cluster(self, evento_id: int, nome: str, dt_ini: str, dt_fim: str, tot_mat: int, tot_veic: int, eixo: str, div_sent: float) -> None:
        con = self.obter_conexao()
        cur = con.cursor()
        cur.execute("""
            INSERT INTO clusters_eventos (id, nome_evento, data_inicio, data_fim, total_materias, total_veiculos, eixo_predominante, divergencia_sentimento)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                nome_evento=excluded.nome_evento,
                data_inicio=excluded.data_inicio,
                data_fim=excluded.data_fim,
                total_materias=excluded.total_materias,
                total_veiculos=excluded.total_veiculos,
                eixo_predominante=excluded.eixo_predominante,
                divergencia_sentimento=excluded.divergencia_sentimento;
        """, (evento_id, nome, dt_ini, dt_fim, tot_mat, tot_veic, eixo, div_sent))
        con.commit()
        con.close()

    def salvar_resultado_estatistico(self, pergunta: str, chave: str, valor: Optional[float], detalhes: Dict[str, Any]) -> None:
        con = self.obter_conexao()
        cur = con.cursor()
        cur.execute("""
            INSERT INTO resultados_estatisticos (pergunta, chave_metrica, valor_numerico, detalhes_json)
            VALUES (?, ?, ?, ?)
        """, (pergunta, chave, valor, json.dumps(detalhes, ensure_ascii=False)))
        con.commit()
        con.close()

    def obter_materias_para_nlp(self) -> List[Dict[str, Any]]:
        con = self.obter_conexao()
        cur = con.cursor()
        cur.execute("""
            SELECT id, titulo, corpo, resumo, data_publicacao, veiculo_id, eixo_tematico
            FROM materias
            WHERE (score_clickbait = 0.0 AND sentimento_polaridade = 0.0 AND sentimento_neu = 0.0)
               OR id NOT IN (SELECT DISTINCT materia_id FROM entidades_mencoes)
            ORDER BY id ASC
        """)
        linhas = [dict(row) for row in cur.fetchall()]
        con.close()
        return linhas

    def obter_materias_sem_embedding(self) -> List[Dict[str, Any]]:
        con = self.obter_conexao()
        cur = con.cursor()
        cur.execute("""
            SELECT m.id, m.titulo, m.corpo
            FROM materias m
            LEFT JOIN embeddings e ON m.id = e.materia_id
            WHERE e.materia_id IS NULL AND m.tamanho_texto >= 150
            ORDER BY m.id ASC
        """)
        linhas = [dict(row) for row in cur.fetchall()]
        con.close()
        return linhas

    def obter_todas_materias_com_embedding(self) -> List[Dict[str, Any]]:
        con = self.obter_conexao()
        cur = con.cursor()
        cur.execute("""
            SELECT m.id, m.veiculo_id, m.titulo, m.corpo, m.data_publicacao, m.ano, m.ano_eleitoral,
                   m.eixo_tematico, m.score_clickbait, m.sentimento_polaridade, m.sentimento_pos,
                   m.sentimento_neg, m.sentimento_neu, m.evento_id, v.codigo as veiculo_codigo,
                   v.nome as veiculo_nome, v.tipo as veiculo_tipo, e.vetor
            FROM materias m
            JOIN veiculos v ON m.veiculo_id = v.id
            JOIN embeddings e ON m.id = e.materia_id
            ORDER BY m.data_publicacao ASC
        """)
        linhas = [dict(row) for row in cur.fetchall()]
        con.close()
        return linhas

    def obter_estatisticas_gerais(self) -> Dict[str, Any]:
        con = self.obter_conexao()
        cur = con.cursor()

        cur.execute("SELECT COUNT(*) FROM urls_coletadas")
        total_urls = cur.fetchone()[0]

        cur.execute("""
            SELECT 
                SUM(CASE WHEN fonte_coleta LIKE '%cdx%' THEN 1 ELSE 0 END),
                SUM(CASE WHEN fonte_coleta NOT LIKE '%cdx%' THEN 1 ELSE 0 END)
            FROM urls_coletadas
        """)
        row_fontes = cur.fetchone()
        total_urls_cdx = row_fontes[0] or 0
        total_urls_sitemaps = row_fontes[1] or 0

        cur.execute("SELECT status_extracao, COUNT(*) FROM urls_coletadas GROUP BY status_extracao")
        status_urls = {row[0]: row[1] for row in cur.fetchall()}

        cur.execute("SELECT COUNT(*) FROM materias")
        total_materias = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM embeddings")
        total_embeddings = cur.fetchone()[0]

        cur.execute("SELECT COUNT(DISTINCT evento_id) FROM materias WHERE evento_id > -1")
        total_eventos = cur.fetchone()[0]

        cur.execute("""
            SELECT v.id, v.nome, v.codigo, v.tipo,
                   (SELECT COUNT(*) FROM materias m WHERE m.veiculo_id = v.id) as total_materias
            FROM veiculos v
            ORDER BY v.tipo, v.nome
        """)
        veiculos_base = [dict(row) for row in cur.fetchall()]

        cur.execute("""
            SELECT 
                v.nome,
                CASE 
                    WHEN u.data_prevista LIKE '201%' OR u.data_prevista LIKE '202%' THEN substr(u.data_prevista, 1, 4)
                    WHEN u.timestamp_cdx LIKE '201%' OR u.timestamp_cdx LIKE '202%' THEN substr(u.timestamp_cdx, 1, 4)
                    WHEN u.url LIKE '%/2015/%' THEN '2015'
                    WHEN u.url LIKE '%/2016/%' THEN '2016'
                    WHEN u.url LIKE '%/2017/%' THEN '2017'
                    WHEN u.url LIKE '%/2018/%' THEN '2018'
                    WHEN u.url LIKE '%/2019/%' THEN '2019'
                    WHEN u.url LIKE '%/2020/%' THEN '2020'
                    WHEN u.url LIKE '%/2021/%' THEN '2021'
                    WHEN u.url LIKE '%/2022/%' THEN '2022'
                    WHEN u.url LIKE '%/2023/%' THEN '2023'
                    WHEN u.url LIKE '%/2024/%' THEN '2024'
                    WHEN u.url LIKE '%/2025/%' THEN '2025'
                    ELSE 'Indefinido'
                END as ano,
                SUM(CASE WHEN u.fonte_coleta LIKE '%cdx%' THEN 1 ELSE 0 END) as cdx,
                SUM(CASE WHEN u.fonte_coleta NOT LIKE '%cdx%' AND u.id IS NOT NULL THEN 1 ELSE 0 END) as sitemap,
                COUNT(u.id) as total
            FROM veiculos v
            LEFT JOIN urls_coletadas u ON v.id = u.veiculo_id
            GROUP BY v.nome, ano
        """)
        agregados = cur.fetchall()

        veiculos_map = {}
        for vb in veiculos_base:
            veiculos_map[vb["nome"]] = {
                "id": vb["id"],
                "nome": vb["nome"],
                "codigo": vb["codigo"],
                "tipo": vb["tipo"],
                "total_materias": vb["total_materias"],
                "total_urls": 0,
                "urls_cdx": 0,
                "urls_sitemaps": 0,
                "anos": {}
            }

        anos_stats = {}
        for row in agregados:
            nome_v = row[0]
            ano = str(row[1]) if row[1] else "Indefinido"
            cdx_cnt = row[2] or 0
            sm_cnt = row[3] or 0
            tot_cnt = row[4] or 0

            if nome_v in veiculos_map:
                veiculos_map[nome_v]["total_urls"] += tot_cnt
                veiculos_map[nome_v]["urls_cdx"] += cdx_cnt
                veiculos_map[nome_v]["urls_sitemaps"] += sm_cnt
                if tot_cnt > 0:
                    veiculos_map[nome_v]["anos"][ano] = tot_cnt

            if ano not in anos_stats:
                anos_stats[ano] = {"cdx": 0, "sitemaps": 0, "total_urls": 0, "materias": 0}
            anos_stats[ano]["cdx"] += cdx_cnt
            anos_stats[ano]["sitemaps"] += sm_cnt
            anos_stats[ano]["total_urls"] += tot_cnt

        cur.execute("""
            SELECT ano, COUNT(*)
            FROM materias
            GROUP BY ano
            ORDER BY ano ASC
        """)
        materias_por_ano = {str(row[0]): row[1] for row in cur.fetchall()}
        for a, m_cnt in materias_por_ano.items():
            if a not in anos_stats:
                anos_stats[a] = {"cdx": 0, "sitemaps": 0, "total_urls": 0, "materias": 0}
            anos_stats[a]["materias"] = m_cnt

        cur.execute("""
            SELECT 
                COALESCE(eixo_tematico, 'Indefinido') as eixo,
                SUM(CASE WHEN fonte_coleta LIKE '%cdx%' THEN 1 ELSE 0 END) as cdx,
                SUM(CASE WHEN fonte_coleta NOT LIKE '%cdx%' THEN 1 ELSE 0 END) as sitemap,
                COUNT(*) as total
            FROM urls_coletadas
            GROUP BY eixo
            ORDER BY total DESC
        """)
        eixos_geral_rows = cur.fetchall()
        eixos_stats = {}
        for r in eixos_geral_rows:
            eixos_stats[r[0]] = {
                "cdx": r[1] or 0,
                "sitemaps": r[2] or 0,
                "total": r[3] or 0,
                "materias": 0
            }

        cur.execute("""
            SELECT COALESCE(eixo_tematico, 'Indefinido') as eixo, COUNT(*)
            FROM materias
            GROUP BY eixo
        """)
        for r_mat in cur.fetchall():
            e_k = r_mat[0]
            if e_k in eixos_stats:
                eixos_stats[e_k]["materias"] = r_mat[1]
            else:
                eixos_stats[e_k] = {"cdx": 0, "sitemaps": 0, "total": 0, "materias": r_mat[1]}

        cur.execute("""
            SELECT 
                v.nome,
                COALESCE(u.eixo_tematico, 'Indefinido') as eixo,
                COUNT(u.id) as total
            FROM veiculos v
            LEFT JOIN urls_coletadas u ON v.id = u.veiculo_id
            GROUP BY v.nome, eixo
        """)
        for r_veic_eixo in cur.fetchall():
            nome_v = r_veic_eixo[0]
            eixo_k = r_veic_eixo[1]
            cnt_k = r_veic_eixo[2] or 0
            if nome_v in veiculos_map:
                if "eixos" not in veiculos_map[nome_v]:
                    veiculos_map[nome_v]["eixos"] = {}
                if cnt_k > 0:
                    veiculos_map[nome_v]["eixos"][eixo_k] = cnt_k

        veiculos_stats = list(veiculos_map.values())
        materias_por_veiculo = {v["nome"]: v["total_materias"] for v in veiculos_stats}
        urls_por_veiculo = {v["nome"]: v["total_urls"] for v in veiculos_stats}
        urls_por_ano = {a: d["total_urls"] for a, d in anos_stats.items()}

        con.close()
        return {
            "total_urls": total_urls,
            "total_urls_cdx": total_urls_cdx,
            "total_urls_sitemaps": total_urls_sitemaps,
            "status_urls": status_urls,
            "total_materias": total_materias,
            "total_embeddings": total_embeddings,
            "total_eventos": total_eventos,
            "veiculos_stats": veiculos_stats,
            "anos_stats": anos_stats,
            "eixos_stats": eixos_stats,
            "materias_por_veiculo": materias_por_veiculo,
            "urls_por_veiculo": urls_por_veiculo,
            "materias_por_ano": materias_por_ano,
            "urls_por_ano": urls_por_ano,
        }
