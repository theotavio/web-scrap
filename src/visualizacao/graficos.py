from pathlib import Path
from typing import Any, Dict, List
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.banco import BancoDados


class GeradorGraficos:
    def __init__(self, banco: BancoDados):
        self.banco = banco
        self.config = obter_configuracao()
        self.logger = obter_logger("visualizacao")
        self.dir_graficos = Path(self.config.obter("geral.diretorio_relatorios", "relatorios")) / "graficos"
        self.dir_graficos.mkdir(parents=True, exist_ok=True)

    def _obter_dataframe_materias(self) -> pd.DataFrame:
        con = self.banco.obter_conexao()
        df = pd.read_sql_query("""
            SELECT m.id, m.titulo, m.data_publicacao, m.ano, m.mes, m.ano_eleitoral,
                   m.eixo_tematico, m.score_clickbait, m.sentimento_polaridade,
                   m.sentimento_pos, m.sentimento_neg, m.sentimento_neu, m.evento_id,
                   v.nome as veiculo_nome, v.tipo as veiculo_tipo, v.codigo as veiculo_codigo
            FROM materias m
            JOIN veiculos v ON m.veiculo_id = v.id
        """, con)
        con.close()
        if not df.empty:
            df["grupo_editorial"] = df["veiculo_tipo"].map({
                "tradicional": "Grupo Tradicional",
                "digital_nativo": "Grupo Digital Nativo"
            }).fillna("Grupo Não Classificado")

            mapa_anon = {}
            trad_idx = 1
            dig_idx = 1
            for v_cod in sorted(df["veiculo_codigo"].unique()):
                sub = df[df["veiculo_codigo"] == v_cod]
                tipo = sub["veiculo_tipo"].iloc[0]
                if tipo == "tradicional":
                    mapa_anon[v_cod] = f"Tradicional {trad_idx}"
                    trad_idx += 1
                else:
                    mapa_anon[v_cod] = f"Digital {dig_idx}"
                    dig_idx += 1
            df["veiculo_anonimizado"] = df["veiculo_codigo"].map(mapa_anon)
        return df

    def gerar_grafico_sentimento_veiculos(self, df: pd.DataFrame) -> str:
        caminho = self.dir_graficos / "distribuicao_sentimento_veiculos.html"
        if df.empty:
            return str(caminho)

        fig = px.box(
            df,
            x="veiculo_anonimizado",
            y="sentimento_polaridade",
            color="grupo_editorial",
            title="Distribuição de Polaridade de Sentimento por Veículo Anonimizado",
            labels={"veiculo_anonimizado": "Veículo", "sentimento_polaridade": "Polaridade de Sentimento [-1 a 1]", "grupo_editorial": "Grupo"},
            template="plotly_white",
            color_discrete_map={"Grupo Tradicional": "#1f77b4", "Grupo Digital Nativo": "#ff7f0e"}
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="gray", annotation_text="Neutro")
        fig.update_layout(xaxis_tickangle=-30, margin=dict(l=40, r=40, t=60, b=80))
        fig.write_html(str(caminho))
        return str(caminho)

    def gerar_grafico_clickbait(self, df: pd.DataFrame) -> str:
        caminho = self.dir_graficos / "clickbait_por_tipo_veiculo.html"
        if df.empty:
            return str(caminho)

        fig = px.box(
            df,
            x="grupo_editorial",
            y="score_clickbait",
            color="grupo_editorial",
            points="all",
            title="Comparativo de Índice de Clickbait: Grupo Tradicional vs. Grupo Digital Nativo",
            labels={"grupo_editorial": "Grupo Editorial", "score_clickbait": "Índice de Clickbait [0 a 1]"},
            template="plotly_white",
            color_discrete_map={"Grupo Tradicional": "#1f77b4", "Grupo Digital Nativo": "#ff7f0e"}
        )
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=60))
        fig.write_html(str(caminho))
        return str(caminho)

    def gerar_grafico_evolucao_eleicoes(self, df: pd.DataFrame) -> str:
        caminho = self.dir_graficos / "evolucao_temporal_vies_eleicoes.html"
        if df.empty:
            return str(caminho)

        df["indice_vies"] = df["sentimento_polaridade"].abs() + (0.5 * df["score_clickbait"])
        df_ano = df.groupby(["ano", "ano_eleitoral", "grupo_editorial"])["indice_vies"].mean().reset_index()

        fig = px.line(
            df_ano,
            x="ano",
            y="indice_vies",
            color="grupo_editorial",
            markers=True,
            title="Evolução Temporal do Índice de Viés por Grupo Editorial nos Ciclos Eleitorais",
            labels={"ano": "Ano", "indice_vies": "Índice Composto (|Sentimento| + 0.5 * Clickbait)", "grupo_editorial": "Grupo"},
            template="plotly_white",
            color_discrete_map={"Grupo Tradicional": "#1f77b4", "Grupo Digital Nativo": "#ff7f0e"}
        )

        anos_eleitorais = self.config.obter("geral.anos_eleitorais", [2016, 2018, 2020, 2022, 2024])
        for ano in anos_eleitorais:
            fig.add_vrect(
                x0=ano - 0.4,
                x1=ano + 0.4,
                fillcolor="rgba(255, 0, 0, 0.12)",
                layer="below",
                line_width=0,
                annotation_text=f"Eleição {ano}",
                annotation_position="top left"
            )

        fig.update_layout(margin=dict(l=40, r=40, t=60, b=60))
        fig.write_html(str(caminho))
        return str(caminho)

    def gerar_grafico_radar(self, df: pd.DataFrame) -> str:
        caminho = self.dir_graficos / "radar_enquadramento_editorial.html"
        if df.empty:
            return str(caminho)

        metricas_grupo = df.groupby("grupo_editorial").agg({
            "score_clickbait": "mean",
            "sentimento_pos": "mean",
            "sentimento_neg": "mean",
            "sentimento_neu": "mean"
        }).reset_index()

        categorias = ["Clickbait", "Positividade", "Negatividade", "Neutralidade"]
        fig = go.Figure()

        cores = {"Grupo Tradicional": "#1f77b4", "Grupo Digital Nativo": "#ff7f0e"}
        for _, row in metricas_grupo.iterrows():
            grp = row["grupo_editorial"]
            valores = [
                row["score_clickbait"],
                row["sentimento_pos"],
                row["sentimento_neg"],
                row["sentimento_neu"]
            ]
            valores.append(valores[0])
            fig.add_trace(go.Scatterpolar(
                r=valores,
                theta=categorias + [categorias[0]],
                fill="toself",
                name=grp,
                line=dict(color=cores.get(grp, "#333"))
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Radar Multidimensional de Enquadramento Editorial: Tradicional vs. Digital Nativo",
            template="plotly_white",
            margin=dict(l=60, r=60, t=60, b=60)
        )
        fig.write_html(str(caminho))
        return str(caminho)

    def gerar_grafico_efeito_mudo_tematico(self, df: pd.DataFrame) -> str:
        caminho = self.dir_graficos / "efeito_mudo_por_area_tematica.html"
        if df.empty:
            return str(caminho)

        df_valid = df[df["eixo_tematico"].notna() & (df["eixo_tematico"] != "")].copy()
        if df_valid.empty:
            return str(caminho)

        contagens = df_valid.groupby(["grupo_editorial", "eixo_tematico"]).size().reset_index(name="contagem")
        totais = df_valid.groupby("grupo_editorial").size().to_dict()
        contagens["porcentagem"] = contagens.apply(
            lambda r: round((r["contagem"] / totais.get(r["grupo_editorial"], 1)) * 100, 2), axis=1
        )

        fig = px.bar(
            contagens,
            x="eixo_tematico",
            y="porcentagem",
            color="grupo_editorial",
            barmode="group",
            title="Efeito Mudo e Silenciamento Seletivo: Distribuição Percentual por Área Temática",
            labels={"eixo_tematico": "Eixo Temático", "porcentagem": "Proporção de Cobertura (%)", "grupo_editorial": "Grupo"},
            template="plotly_white",
            color_discrete_map={"Grupo Tradicional": "#1f77b4", "Grupo Digital Nativo": "#ff7f0e"}
        )
        fig.update_layout(margin=dict(l=40, r=40, t=60, b=60))
        fig.write_html(str(caminho))
        return str(caminho)

    def gerar_grafico_rede(self, G_proj: nx.Graph) -> str:
        caminho = self.dir_graficos / "rede_similaridade_interativa.html"
        if G_proj.number_of_nodes() == 0:
            return str(caminho)

        pos = nx.spring_layout(G_proj, seed=42, weight="weight")

        edge_x = []
        edge_y = []

        for edge in G_proj.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1.5, color="#888"),
            hoverinfo="none",
            mode="lines"
        )

        node_x = []
        node_y = []
        node_text = []
        node_color = []

        for node in G_proj.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            rotulo = node[1].get("rotulo", node[0])
            tipo = node[1].get("tipo_editorial", "padrao")
            node_text.append(f"{rotulo} ({tipo})")
            node_color.append("#1f77b4" if tipo == "tradicional" else "#ff7f0e")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=[node[1].get("rotulo", node[0]) for node in G_proj.nodes(data=True)],
            textposition="top center",
            hoverinfo="text",
            hovertext=node_text,
            marker=dict(
                size=22,
                color=node_color,
                line=dict(width=2, color="#333")
            )
        )

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title="Rede de Proximidade Editorial entre Veículos Anonimizados",
                showlegend=False,
                hovermode="closest",
                template="plotly_white",
                margin=dict(b=40, l=40, r=40, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
        )
        fig.write_html(str(caminho))
        return str(caminho)

    def executar(self, G_proj: nx.Graph) -> Dict[str, str]:
        df = self._obter_dataframe_materias()
        caminhos = {
            "sentimento": self.gerar_grafico_sentimento_veiculos(df),
            "clickbait": self.gerar_grafico_clickbait(df),
            "eleicoes": self.gerar_grafico_evolucao_eleicoes(df),
            "radar": self.gerar_grafico_radar(df),
            "efeito_mudo": self.gerar_grafico_efeito_mudo_tematico(df),
            "rede": self.gerar_grafico_rede(G_proj)
        }
        self.logger.info(f"Graficos interativos gerados: {caminhos}")
        return caminhos
