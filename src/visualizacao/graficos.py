import io
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.banco import BancoDados


class GeradorGraficosArtigo:
    def __init__(
        self,
        dir_dados_graficos: Optional[str] = None,
        dir_dados_grafos: Optional[str] = None,
        dir_saida_graficos: str = "resultados/graficos",
        dir_saida_grafos_imagens: str = "resultados/grafos/imagens"
    ):
        self.dir_dados_graficos = Path(dir_dados_graficos) if dir_dados_graficos else None
        self.dir_dados_grafos = Path(dir_dados_grafos) if dir_dados_grafos else None
        self.dir_saida_graficos = Path(dir_saida_graficos)
        self.dir_saida_grafos_imagens = Path(dir_saida_grafos_imagens)

        self.dir_saida_graficos.mkdir(parents=True, exist_ok=True)
        self.dir_saida_grafos_imagens.mkdir(parents=True, exist_ok=True)

        self.cores_eixos = {
            "Politico": "#8B5CF6",
            "Economico": "#EAB308",
            "Social": "#EC4899",
            "Ambiental": "#10B981",
            "Internacional": "#06B6D4",
            "Esportivo": "#E11D48"
        }
        self.cor_tradicional = "#1E3A8A"
        self.cor_digital = "#EA580C"

    def _obter_dataframe(self, nome_csv: str) -> pd.DataFrame:
        if self.dir_dados_graficos:
            pasta = Path(self.dir_dados_graficos)
            if not pasta.exists():
                raise FileNotFoundError(f"Diretório informado não existe: {self.dir_dados_graficos}")
            p_csv = pasta / nome_csv
            if p_csv.exists():
                return pd.read_csv(p_csv)
            p_zip = pasta / "dados_graficos_csv.zip"
            if p_zip.exists():
                with zipfile.ZipFile(p_zip, "r") as zf:
                    if nome_csv in zf.namelist():
                        return pd.read_csv(zf.open(nome_csv))
            raise FileNotFoundError(f"Arquivo {nome_csv} não encontrado no diretório informado: {self.dir_dados_graficos}")

        pastas_candidatas = [
            Path("resultados/dados_graficos"),
            Path("banco_dados")
        ]
        for pasta in pastas_candidatas:
            if not pasta.exists():
                continue
            p_csv = pasta / nome_csv
            if p_csv.exists():
                return pd.read_csv(p_csv)
            p_zip = pasta / "dados_graficos_csv.zip"
            if p_zip.exists():
                with zipfile.ZipFile(p_zip, "r") as zf:
                    if nome_csv in zf.namelist():
                        return pd.read_csv(zf.open(nome_csv))
        raise FileNotFoundError(f"Arquivo {nome_csv} não encontrado em resultados/dados_graficos.")

    def gerar_01_balanco_bipolar_sentimento(self) -> str:
        df = self._obter_dataframe("01_sentimento_ridgeline_densidade.csv")
        caminho_png = self.dir_saida_graficos / "01_balanco_bipolar_sentimento.png"

        fig, axes = plt.subplots(1, 2, figsize=(16, 9.8), dpi=300, sharex=True)
        fig.patch.set_facecolor("#FAFAFA")

        trad_veiculos = ["Tradicional 1", "Tradicional 2", "Tradicional 3", "Tradicional 4", "Tradicional 5"]
        dig_veiculos = ["Digital 1", "Digital 2", "Digital 3", "Digital 4", "Digital 5"]

        def plot_panel(ax, veiculos, titulo, cor_primaria):
            ax.set_facecolor("#FFFFFF")
            ax.grid(axis="x", color="#E2E8F0", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_axisbelow(True)
            ax.axvline(0, color="#64748B", linestyle="-", linewidth=1.2, alpha=0.8)

            y_positions = np.arange(len(veiculos))[::-1]

            for idx, v in enumerate(veiculos):
                sub = df[df["veiculo"] == v].sort_values("polaridade")
                pols = sub["polaridade"].values
                dens = sub["densidade"].values

                tot_area = np.trapezoid(dens, pols)
                mean_pol = np.trapezoid(dens * pols, pols) / tot_area
                neg_pct = (np.trapezoid(np.where(pols < -0.05, dens, 0), pols) / tot_area) * 100
                pos_pct = (np.trapezoid(np.where(pols > 0.05, dens, 0), pols) / tot_area) * 100
                neu_pct = 100.0 - neg_pct - pos_pct

                y_base = y_positions[idx]
                norm_dens = dens / 4.8 * 0.38

                mask_neg = pols <= 0
                mask_pos = pols >= 0

                ax.fill_between(pols[mask_neg], y_base, y_base + norm_dens[mask_neg], color="#EF4444", alpha=0.65, linewidth=0)
                ax.fill_between(pols[mask_pos], y_base, y_base + norm_dens[mask_pos], color=cor_primaria, alpha=0.65, linewidth=0)
                ax.plot(pols, y_base + norm_dens, color="#1E293B", linewidth=1.1)
                ax.axhline(y_base, color="#CBD5E1", linewidth=0.8)

                ax.scatter(mean_pol, y_base + 0.04, color="#0F172A", s=35, zorder=5, edgecolor="#FFFFFF", linewidth=1)
                ax.text(mean_pol, y_base + 0.12, f"{mean_pol:+.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#0F172A")

                pct_label = f"Pos: {pos_pct:.0f}% | Neu: {neu_pct:.0f}% | Neg: {neg_pct:.0f}%"
                ax.text(0.78, y_base + 0.32, pct_label, ha="right", va="center", fontsize=7.5, color="#475569",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#F1F5F9", edgecolor="none", alpha=0.9))

            ax.set_yticks(y_positions)
            ax.set_yticklabels(veiculos, fontsize=10.5, fontweight="bold", color="#1E293B")
            ax.set_title(titulo, fontsize=12.5, fontweight="bold", color="#0F172A", pad=15)
            ax.set_xlabel("Polaridade de Sentimento (-1.0 Crítico a +1.0 Favorável)", fontsize=9.5, fontweight="bold", color="#334155")
            ax.set_xlim(-0.85, 0.85)

        plot_panel(axes[0], trad_veiculos, "A. Bloco Tradicional (Consenso Institucional)", "#2563EB")
        plot_panel(axes[1], dig_veiculos, "B. Bloco Digital Nativo (Bifurcação e Crítica)", "#D97706")

        trad_patch = patches.Patch(facecolor="#2563EB", alpha=0.65, label="Sentimento Favorável (Imprensa Tradicional)")
        dig_patch = patches.Patch(facecolor="#D97706", alpha=0.65, label="Sentimento Favorável (Digital Nativo)")
        neg_patch = patches.Patch(facecolor="#EF4444", alpha=0.65, label="Sentimento Crítico / Negativo (Polaridade < 0)")
        mean_line = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#0F172A", markersize=7, label="Média Ponderada de Sentimento")
        fig.legend(handles=[trad_patch, dig_patch, neg_patch, mean_line], loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=4, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

        fig.suptitle("Distribuição e Balanço Bipolar de Sentimento Editorial (2015–2025)", fontsize=15, fontweight="bold", color="#0F172A", y=0.98)
        plt.subplots_adjust(top=0.91, bottom=0.15, left=0.12, right=0.97, wspace=0.20)
        plt.savefig(str(caminho_png), dpi=300)
        plt.close()
        return str(caminho_png)

    def gerar_02_lollipop_clickbait(self) -> str:
        df = self._obter_dataframe("02_clickbait_ranking_lollipop.csv")
        caminho_png = self.dir_saida_graficos / "02_lollipop_ranking_clickbait.png"
        df_sorted = df.sort_values("clickbait_medio", ascending=True).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(14, 8.2), dpi=300)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="x", color="#E2E8F0", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)

        media_nac = 0.0607
        ax.axvline(media_nac, color="#DC2626", linestyle="--", linewidth=1.5, zorder=2)
        ax.text(media_nac, len(df_sorted) - 0.1, f"Média Nacional: {media_nac:.4f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#DC2626",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF2F2", edgecolor="#FECACA", alpha=0.95))

        y_pos = np.arange(len(df_sorted))
        for idx, (_, row) in enumerate(df_sorted.iterrows()):
            v = row["veiculo"]
            val = float(row["clickbait_medio"])
            p90 = float(row["percentil_90"])
            is_trad = "Tradicional" in v
            cor = self.cor_tradicional if is_trad else self.cor_digital

            ax.plot([0, val], [idx, idx], color=cor, linewidth=3.0, zorder=3)
            ax.plot([val, p90], [idx, idx], color="#94A3B8", linewidth=1.4, linestyle=":", zorder=3)

            ax.scatter(val, idx, color=cor, s=90, zorder=4, edgecolor="#0F172A", linewidth=1.0)
            ax.text(val, idx + 0.18, f"{val:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=cor)

            ax.scatter(p90, idx, color="#64748B", s=45, marker="D", zorder=4)
            ax.text(p90 + 0.002, idx, f"P90: {p90:.3f}", ha="left", va="center", fontsize=8.0, color="#64748B")

        trad_line = plt.Line2D([0], [0], color=self.cor_tradicional, marker="o", lw=2.5, markersize=7, label="Mídia Tradicional (Média)")
        dig_line = plt.Line2D([0], [0], color=self.cor_digital, marker="o", lw=2.5, markersize=7, label="Mídia Digital Nativa (Média)")
        p90_marker = plt.Line2D([0], [0], color="#64748B", marker="D", lw=0, markersize=6, label="Percentil P90 (Dispersão Extrema)")
        ref_line = plt.Line2D([0], [0], color="#DC2626", linestyle="--", lw=1.5, label=f"Média Nacional Referência ({media_nac:.4f})")
        ax.legend(handles=[trad_line, dig_line, p90_marker, ref_line], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted["veiculo"], fontsize=10.5, fontweight="bold", color="#1E293B")
        ax.set_xlim(0, 0.215)
        ax.set_ylim(-0.6, len(df_sorted) + 0.5)
        ax.set_xlabel("Score de Sensacionalismo / Clickbait Médio (0 a 1)", fontsize=10, fontweight="bold", color="#334155", loc="center")
        ax.set_title("Ranking Editorial de Sensacionalismo Textual e Clickbait (2015-2025)",
                     fontsize=13, fontweight="bold", color="#0F172A", pad=15)

        plt.subplots_adjust(top=0.92, bottom=0.18, left=0.12, right=0.96)
        plt.savefig(str(caminho_png), dpi=300)
        plt.close()
        return str(caminho_png)

    def gerar_03a_cardiograma_ciclos_eleitorais(self) -> str:
        df3 = self._obter_dataframe("03_ciclos_eleitorais_streamgraph_heatmap.csv")
        caminho_png = self.dir_saida_graficos / "03a_cardiograma_ciclos_eleitorais.png"
        df3["data"] = pd.to_datetime(df3["ano_mes"] + "-01")

        fig, ax = plt.subplots(figsize=(15, 7.8), dpi=300)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, color="#F1F5F9", linestyle="-", linewidth=0.8)

        ax.plot(df3["data"], df3["polaridade_tradicional"], color=self.cor_tradicional, linewidth=2.4, label="Polaridade Mídia Tradicional", zorder=4)
        ax.plot(df3["data"], df3["polaridade_digital"], color=self.cor_digital, linewidth=2.4, label="Polaridade Mídia Digital Nativa", zorder=4)

        ax.fill_between(df3["data"], df3["polaridade_tradicional"], df3["polaridade_digital"], color="#DC2626", alpha=0.18, label="Fosso de Divergência Editorial", zorder=3)

        eleicoes = [
            (2016, "2016: Eleições Municipais\n(Pós-Impeachment)"),
            (2018, "2018: Eleições Presidenciais\n(Pico de Ruptura)"),
            (2020, "2020: Eleições Municipais\n(Pandemia COVID-19)"),
            (2022, "2022: Eleições Presidenciais\n(Clímax Bipolar)"),
            (2024, "2024: Eleições Municipais\n(Consolidação Bipolar)")
        ]

        for ano_el, label in eleicoes:
            dt_inicio = pd.to_datetime(f"{ano_el}-01-01")
            dt_fim = pd.to_datetime(f"{ano_el}-12-31")
            ax.axvspan(dt_inicio, dt_fim, color="#FEF08A", alpha=0.30, zorder=1)
            ax.axvline(dt_inicio, color="#B45309", linestyle=":", linewidth=2.2, zorder=5)
            ax.text(pd.to_datetime(f"{ano_el}-07-01"), 0.44, label, ha="center", va="top", fontsize=8.0, fontweight="bold", color="#854D0E",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#FEF9C3", edgecolor="#FDE047", alpha=0.9), zorder=6)

        ax.set_ylabel("Polaridade de Sentimento (-1 a +1)", fontsize=10, fontweight="bold", color="#334155")
        ax.set_xlabel("Linha do Tempo (Ano a Ano)", fontsize=10, fontweight="bold", color="#334155")
        ax.set_ylim(-0.15, 0.49)
        ax.set_xlim(pd.to_datetime("2014-12-01"), pd.to_datetime("2025-12-31"))
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        line_trad = plt.Line2D([0], [0], color=self.cor_tradicional, linewidth=2.4, label="Polaridade Mídia Tradicional")
        line_dig = plt.Line2D([0], [0], color=self.cor_digital, linewidth=2.4, label="Polaridade Mídia Digital Nativa")
        patch_fosso = patches.Patch(facecolor="#DC2626", alpha=0.18, label="Fosso de Divergência Editorial")
        line_eleicao = plt.Line2D([0], [0], color="#B45309", linestyle=":", linewidth=2.2, label="Linha Pontilhada: Marco de Ano Eleitoral")
        ax.legend(handles=[line_trad, line_dig, patch_fosso, line_eleicao], loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

        fig.suptitle("Cardiograma Temporal da Divergência Editorial nos Ciclos Eleitorais (2015–2025)",
                     fontsize=13, fontweight="bold", color="#0F172A", y=0.97)

        plt.subplots_adjust(top=0.90, bottom=0.18, left=0.06, right=0.96)
        plt.savefig(str(caminho_png), dpi=300)
        plt.close()
        return str(caminho_png)

    def gerar_04_sankey_enquadramento(self) -> str:
        df = self._obter_dataframe("04_enquadramento_editorial_sankey.csv")
        caminho_png = self.dir_saida_graficos / "04_sankey_enquadramento_editorial.png"

        grupos = ["Grupo Tradicional", "Grupo Digital Nativo"]
        eixos = sorted(df["eixo_tematico"].unique())
        tons = ["Sentimento Positivo", "Sentimento Neutro", "Sentimento Negativo"]

        labels_todos = grupos + eixos + tons
        mapa_id = {nome: i for i, nome in enumerate(labels_todos)}

        cores_nos = []
        for nome in labels_todos:
            if "Tradicional" in nome:
                cores_nos.append(self.cor_tradicional)
            elif "Digital" in nome:
                cores_nos.append(self.cor_digital)
            elif nome in self.cores_eixos:
                cores_nos.append(self.cores_eixos[nome])
            elif "Positivo" in nome:
                cores_nos.append("#10B981")
            elif "Neutro" in nome:
                cores_nos.append("#94A3B8")
            elif "Negativo" in nome:
                cores_nos.append("#EF4444")
            else:
                cores_nos.append("#64748B")

        fontes = []
        alvos = []
        valores = []
        cores_links = []

        g1 = df.groupby(["origem_grupo", "eixo_tematico"])["volume_materias"].sum().reset_index()
        for _, r in g1.iterrows():
            fontes.append(mapa_id[r["origem_grupo"]])
            alvos.append(mapa_id[r["eixo_tematico"]])
            valores.append(int(r["volume_materias"]))
            if "Tradicional" in r["origem_grupo"]:
                cores_links.append("rgba(30, 58, 138, 0.35)")
            else:
                cores_links.append("rgba(234, 88, 12, 0.35)")

        g2 = df.groupby(["eixo_tematico", "tom_editorial"])["volume_materias"].sum().reset_index()
        for _, r in g2.iterrows():
            fontes.append(mapa_id[r["eixo_tematico"]])
            alvos.append(mapa_id[r["tom_editorial"]])
            valores.append(int(r["volume_materias"]))
            if "Positivo" in r["tom_editorial"]:
                cores_links.append("rgba(16, 185, 129, 0.35)")
            elif "Neutro" in r["tom_editorial"]:
                cores_links.append("rgba(148, 163, 184, 0.35)")
            else:
                cores_links.append("rgba(239, 68, 68, 0.35)")

        fig = go.Figure()

        fig.add_trace(go.Sankey(
            arrangement="snap",
            node=dict(
                pad=28,
                thickness=22,
                line=dict(color="#1E293B", width=0.8),
                label=labels_todos,
                color=cores_nos
            ),
            link=dict(
                source=fontes,
                target=alvos,
                value=valores,
                color=cores_links
            )
        ))

        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=self.cor_tradicional, symbol="square"), name="Mídia Tradicional"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=self.cor_digital, symbol="square"), name="Mídia Digital Nativa"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color="#10B981", symbol="square"), name="Sentimento Positivo"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color="#94A3B8", symbol="square"), name="Sentimento Neutro"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color="#EF4444", symbol="square"), name="Sentimento Negativo"))

        fig.add_annotation(x=0.01, y=1.04, text="<b>1. Modelo Editorial</b>", showarrow=False, font=dict(size=12, color="#0F172A"))
        fig.add_annotation(x=0.50, y=1.04, text="<b>2. Eixo Temático</b>", showarrow=False, font=dict(size=12, color="#0F172A"))
        fig.add_annotation(x=0.99, y=1.04, text="<b>3. Valência de Sentimento</b>", showarrow=False, font=dict(size=12, color="#0F172A"))

        fig.update_layout(
            title=dict(
                text="<b>Fluxo Editorial Multidimensional — Grupos, Temas e Valências (2015–2025)</b>",
                font=dict(size=14, color="#0F172A"),
                x=0.5,
                xanchor="center",
                y=0.97
            ),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.06,
                xanchor="center",
                x=0.5,
                bgcolor="#FFFFFF",
                bordercolor="#CBD5E1",
                borderwidth=1,
                font=dict(size=11, color="#1E293B")
            ),
            template="plotly_white",
            font=dict(size=11, color="#1E293B"),
            height=750,
            width=1200,
            margin=dict(l=60, r=60, t=90, b=90)
        )

        fig.write_image(str(caminho_png), scale=2.5)
        return str(caminho_png)

    def gerar_05a_radar_silhueta_editorial(self) -> str:
        df5 = self._obter_dataframe("05_efeito_mudo_slopegraph.csv")
        caminho_png = self.dir_saida_graficos / "05a_radar_silhueta_editorial.png"

        categories = df5["eixo_tematico"].tolist()
        N = len(categories)

        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        trad_vals = df5["proporcao_tradicional_pct"].tolist()
        trad_vals += trad_vals[:1]

        dig_vals = df5["proporcao_digital_pct"].tolist()
        dig_vals += dig_vals[:1]

        fig = plt.figure(figsize=(14, 11), dpi=300)
        fig.patch.set_facecolor("#FAFAFA")

        ax = fig.add_subplot(1, 1, 1, polar=True)
        ax.set_facecolor("#FFFFFF")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)

        plt.xticks(angles[:-1], categories, fontsize=11, fontweight="bold", color="#1E293B")
        ax.tick_params(pad=24)

        ax.set_rscale("linear")
        ax.set_rticks([10, 20, 30, 40, 50, 60])
        ax.set_yticklabels(["10%", "20%", "30%", "40%", "50%", "60%"], fontsize=10, fontweight="bold", color="#0F172A")
        ax.grid(color="#94A3B8", linestyle="--", linewidth=1.1, alpha=0.85)
        ax.spines["polar"].set_color("#64748B")
        ax.spines["polar"].set_linewidth(1.3)
        ax.set_rlim(0, 84)

        ax.plot(angles, trad_vals, linewidth=2.8, linestyle="solid", color=self.cor_tradicional, label="Mídia Tradicional")
        ax.fill(angles, trad_vals, color="#3B82F6", alpha=0.35)

        ax.plot(angles, dig_vals, linewidth=2.8, linestyle="solid", color=self.cor_digital, label="Mídia Digital Nativa")
        ax.fill(angles, dig_vals, color="#FB923C", alpha=0.35)

        ax.scatter(angles[:-1], trad_vals[:-1], color=self.cor_tradicional, s=55, zorder=5)
        ax.scatter(angles[:-1], dig_vals[:-1], color=self.cor_digital, s=55, zorder=5)

        for i, cat in enumerate(categories):
            t_v = trad_vals[i]
            d_v = dig_vals[i]
            ang = angles[i]
            max_v = max(t_v, d_v)
            lbl = f"Dig: {d_v:.1f}%\nTrad: {t_v:.1f}%"
            rad_offset = 15.0 if max_v < 40 else 12.0
            ax.annotate(
                lbl,
                xy=(ang, max_v),
                xytext=(ang, max_v + rad_offset),
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#FFFFFF", edgecolor="#94A3B8", linewidth=1.0, alpha=0.96),
                arrowprops=dict(arrowstyle="->", color="#1E293B", linewidth=1.2, shrinkA=3, shrinkB=4),
                zorder=7
            )

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=10.5)

        fig.suptitle("Radar de Silhueta Editorial: Proporção Temática de Cobertura (2015–2025)",
                     fontsize=13.5, fontweight="bold", color="#0F172A", y=0.97)

        plt.subplots_adjust(top=0.90, bottom=0.14, left=0.08, right=0.92)
        plt.savefig(str(caminho_png), dpi=300)
        plt.close()
        return str(caminho_png)

    def gerar_05b_tornado_efeito_mudo(self) -> str:
        df5 = self._obter_dataframe("05_efeito_mudo_slopegraph.csv")
        caminho_png = self.dir_saida_graficos / "05b_tornado_efeito_mudo.png"

        fig, ax = plt.subplots(figsize=(15, 7.2), dpi=300)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="x", color="#E2E8F0", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)

        df5_sorted = df5.sort_values("diferenca_pontos_pct", ascending=True).reset_index(drop=True)
        y_pos = np.arange(len(df5_sorted))

        diffs = df5_sorted["diferenca_pontos_pct"].values
        colors = [self.cor_digital if d < 0 else self.cor_tradicional for d in diffs]

        ax.barh(y_pos, diffs, height=0.55, color=colors, alpha=0.85, edgecolor="#0F172A", linewidth=0.8)
        ax.axvline(0, color="#0F172A", linewidth=1.2)

        for idx, row in df5_sorted.iterrows():
            val = row["diferenca_pontos_pct"]
            if val < -10:
                lbl = f"-{abs(val):.1f} p.p. (1.5x mais no Digital)"
                ax.text(val - 0.8, idx, lbl, ha="right", va="center", fontsize=9.0, fontweight="bold", color="#C2410C")
            elif val > 10:
                lbl = f"+{val:.1f} p.p. (2.3x mais no Tradicional)"
                ax.text(val + 0.8, idx, lbl, ha="left", va="center", fontsize=9.0, fontweight="bold", color="#1E3A8A")
            else:
                align = "left" if val >= 0 else "right"
                offset = 0.5 if val >= 0 else -0.5
                ax.text(val + offset, idx, f"{val:+.1f} p.p. (Equilibrado)", ha=align, va="center", fontsize=8.0, color="#64748B")

        bar_trad = patches.Patch(facecolor=self.cor_tradicional, alpha=0.85, label="Maior Presença na Mídia Tradicional (+p.p.)")
        bar_dig = patches.Patch(facecolor=self.cor_digital, alpha=0.85, label="Maior Presença na Mídia Digital Nativa (-p.p.)")
        ax.legend(handles=[bar_trad, bar_dig], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df5_sorted["eixo_tematico"], fontsize=11, fontweight="bold", color="#1E293B")
        ax.set_xlim(-38, 38)
        ax.set_xlabel("<- Maior Presença no Digital | Diferença em Pontos Percentuais (p.p.) | Maior Presença no Tradicional ->", fontsize=10, fontweight="bold", color="#334155")
        ax.set_title("O Efeito Mudo: Assimetria Temática entre Imprensa Tradicional e Digital Nativa",
                     fontsize=13, fontweight="bold", color="#0F172A", pad=15)

        plt.subplots_adjust(top=0.91, bottom=0.18, left=0.14, right=0.95)
        plt.savefig(str(caminho_png), dpi=300)
        plt.close()
        return str(caminho_png)

    def gerar_06a_serie_temporal_divergencia_tematica(self) -> str:
        df = self._obter_dataframe("08_origem_metastase_serie_historica.csv")
        caminho_png = self.dir_saida_graficos / "06a_serie_temporal_divergencia_tematica.png"
        df["data"] = pd.to_datetime(df["mes_ano"] + "-01")

        fig, ax = plt.subplots(figsize=(15, 7.8), dpi=300)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, color="#F1F5F9", linestyle="-", linewidth=0.8)

        ax.plot(df["data"], df["divergencia_geral"], color="#0F172A", linestyle="--", linewidth=2.8, label="Média da Divergência Editorial", zorder=5)
        ax.plot(df["data"], df["divergencia_politica"], color=self.cores_eixos["Politico"], linewidth=2.6, label="Política", zorder=4)
        ax.plot(df["data"], df["divergencia_economia"], color=self.cores_eixos["Economico"], linewidth=2.0, label="Economia", zorder=3)
        ax.plot(df["data"], df["divergencia_social_saude"], color=self.cores_eixos["Social"], linewidth=2.0, label="Saúde / Social", zorder=3)
        ax.plot(df["data"], df["divergencia_ambiental"], color=self.cores_eixos["Ambiental"], linewidth=2.0, label="Meio Ambiente", zorder=3)
        ax.plot(df["data"], df["divergencia_internacional"], color=self.cores_eixos["Internacional"], linewidth=2.0, label="Internacional", zorder=3)

        picos_eixos = [
            ("2015-08-01", "divergencia_ambiental", "Pico Histórico Ambiental (0.265)\nSeca e Mudanças Climáticas", 18, 32, "left", self.cores_eixos["Ambiental"]),
            ("2015-11-01", "divergencia_internacional", "Pico Histórico Internacional (0.165)\nGeopolítica e Acordo de Paris", 12, -48, "left", self.cores_eixos["Internacional"]),
            ("2016-08-01", "divergencia_politica", "Pico Histórico Política (0.222)\nImpeachment Dilma Rousseff", 0, 42, "center", self.cores_eixos["Politico"]),
            ("2021-04-01", "divergencia_social_saude", "Pico Histórico Saúde/Social (0.187)\nSegunda Onda COVID-19 / CPI", 0, 40, "center", self.cores_eixos["Social"]),
            ("2021-10-01", "divergencia_economia", "Pico Histórico Economia (0.153)\nInflação e Choque de Preços", 0, -48, "center", self.cores_eixos["Economico"])
        ]

        for d_str, col_name, txt, offset_x, offset_y, ha_align, cor_borda in picos_eixos:
            dt = pd.to_datetime(d_str)
            sub_m = df[df["data"] == dt]
            if not sub_m.empty:
                y_pt = float(sub_m[col_name].values[0])
            else:
                y_pt = 0.20

            ax.axvline(dt, color="#CBD5E1", linestyle=":", linewidth=1.0, zorder=2)
            ax.scatter(dt, y_pt, color=cor_borda, s=50, zorder=5, edgecolor="#0F172A", linewidth=0.8)

            va_align = "bottom" if offset_y > 0 else "top"
            ax.annotate(
                txt,
                xy=(dt, y_pt),
                xytext=(offset_x, offset_y),
                textcoords="offset points",
                ha=ha_align,
                va=va_align,
                fontsize=8.5,
                fontweight="bold",
                color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFFFFF", edgecolor=cor_borda, linewidth=1.2, alpha=0.96),
                arrowprops=dict(arrowstyle="->", color=cor_borda, linewidth=1.2),
                zorder=6
            )

        ax.set_ylabel("Divergência Editorial Média (0 a 1)", fontsize=10, fontweight="bold", color="#334155")
        ax.set_xlabel("Linha do Tempo (Ano a Ano)", fontsize=10, fontweight="bold", color="#334155")
        ax.set_ylim(0.04, 0.32)
        ax.set_xlim(pd.to_datetime("2014-12-01"), pd.to_datetime("2025-12-31"))
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=6, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

        fig.suptitle("Série Histórica da Divergência Editorial por Eixo Temático (2015–2025)",
                     fontsize=14, fontweight="bold", color="#0F172A", y=0.97)

        plt.subplots_adjust(top=0.90, bottom=0.18, left=0.06, right=0.96)
        plt.savefig(str(caminho_png), dpi=300)
        plt.close()
        return str(caminho_png)

    def gerar_06b_metastase_tematica_polarizacao(self) -> str:
        caminho_png = self.dir_saida_graficos / "06b_metastase_tematica_polarizacao.png"

        eixos_metastase = ["Meio Ambiente", "Saúde / Sociedade", "Política", "Internacional", "Economia"]
        pre_2018 = [0.1413, 0.1482, 0.1985, 0.1510, 0.1580]
        pos_2018 = [0.1820, 0.1793, 0.2080, 0.1620, 0.1661]
        variacoes = ["+28.8%", "+21.0%", "+4.8%", "+7.3%", "+5.1%"]

        fig, ax = plt.subplots(figsize=(14, 8.4), dpi=300)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FFFFFF")
        ax.grid(axis="x", color="#E2E8F0", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)

        y = np.arange(len(eixos_metastase))
        height = 0.35

        ax.barh(y - height / 2, pre_2018, height=height, label="Período 1: Pré-2018 (2015–2017)", color="#94A3B8", edgecolor="#0F172A", linewidth=0.8)

        cores_pos = [self.cores_eixos["Ambiental"], self.cores_eixos["Social"], self.cores_eixos["Politico"], self.cores_eixos["Internacional"], self.cores_eixos["Economico"]]
        ax.barh(y + height / 2, pos_2018, height=height, label="Período 2: Pós-2018 (2018–2025)", color=cores_pos, edgecolor="#0F172A", linewidth=0.8)

        for i in range(len(eixos_metastase)):
            ax.text(pos_2018[i] + 0.003, y[i] + height / 2, f"Salto: {variacoes[i]}", ha="left", va="center", fontsize=9.0, fontweight="bold", color=cores_pos[i])

        bar_pre = patches.Patch(facecolor="#94A3B8", label="Período 1: Pré-2018 (2015–2017) | Média Basal")
        bar_pos = patches.Patch(facecolor=self.cores_eixos["Politico"], label="Período 2: Pós-2018 (2018–2025) | Expansão nos Eixos")
        salto_marker = plt.Line2D([0], [0], color="#DC2626", marker=r"$\Delta$", lw=0, markersize=8, label="Salto (%): Variação relativa pós-2018")
        ax.legend(handles=[bar_pre, bar_pos, salto_marker], loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

        ax.set_yticks(y)
        ax.set_yticklabels(eixos_metastase, fontsize=11, fontweight="bold", color="#1E293B")
        ax.set_xlim(0, 0.25)
        ax.set_xlabel("Índice Médio de Divergência Editorial de Sentimento", fontsize=10, fontweight="bold", color="#334155")
        ax.set_title("O Fenômeno da Metástase Temática: Comparação da Polarização Pré vs. Pós-2018",
                     fontsize=13, fontweight="bold", color="#0F172A", pad=15)

        plt.subplots_adjust(top=0.91, bottom=0.22, left=0.14, right=0.95)
        plt.savefig(str(caminho_png), dpi=300)
        plt.close()
        return str(caminho_png)

    def gerar_07_regua_betweenness(self) -> str:
        df = self._obter_dataframe("07_regua_polaridade_bubble_motion.csv")
        caminho_png = self.dir_saida_graficos / "07_regua_polaridade_betweenness.png"
        df["data"] = pd.to_datetime(df["data_evento"])

        eixos_validos = ["Politico", "Economico", "Social"]
        df = df[df["eixo_tematico"].isin(eixos_validos)].copy()

        fig, ax = plt.subplots(figsize=(16, 9.2), dpi=300)
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FFFFFF")
        ax.grid(True, color="#F1F5F9", linestyle="-", linewidth=0.8)

        media_div = float(df["divergencia_sentimento"].mean())
        ax.axhline(media_div, color="#DC2626", linestyle="--", linewidth=1.4, label=f"Divergência Média Nacional ({media_div:.4f})", zorder=3)

        nomes_formatados = {
            "Politico": "Eixo Político",
            "Economico": "Eixo Econômico",
            "Social": "Eixo Social"
        }

        max_bc = df["score_fato_ponte_betweenness"].max()
        for eixo in eixos_validos:
            grp = df[df["eixo_tematico"] == eixo]
            cor = self.cores_eixos.get(eixo, "#64748B")
            tams = 25.0 + (grp["score_fato_ponte_betweenness"].values / max_bc) ** 1.6 * 320.0
            ax.scatter(grp["data"], grp["divergencia_sentimento"], s=tams, color=cor, alpha=0.68,
                       edgecolor="#1E293B", linewidth=0.6, label=nomes_formatados[eixo], zorder=4)

        eventos_chave = [
            ("2018-10-06", "Eleição Presidencial 2018 (1º Turno)", "Politico", 60),
            ("2021-09-08", "Crise Institucional de 7 de Setembro", "Politico", 65),
            ("2020-03-17", "Pacote Econômico COVID-19", "Economico", -60),
            ("2022-01-24", "Sanção do Orçamento da União", "Economico", -60),
            ("2021-01-17", "Início da Vacinação COVID-19", "Social", -65),
            ("2021-03-24", "Pico da 2ª Onda COVID & Isolamento", "Social", 65)
        ]

        for dt_str, titulo, eixo_nome, offset_y in eventos_chave:
            dt = pd.to_datetime(dt_str)
            sub_ev = df[(df["data"] == dt) & (df["eixo_tematico"] == eixo_nome)]
            if not sub_ev.empty:
                y_val = float(sub_ev["divergencia_sentimento"].values[0])
            else:
                y_val = 0.30
            cor_eixo = self.cores_eixos.get(eixo_nome, "#1E293B")
            va_align = "bottom" if offset_y > 0 else "top"
            ax.annotate(
                f"Fato-Ponte: {titulo}\n[{dt.strftime('%d/%m/%Y')} | {nomes_formatados[eixo_nome]}]",
                xy=(dt, y_val),
                xytext=(0, offset_y),
                textcoords="offset points",
                ha="center",
                va=va_align,
                fontsize=8.5,
                fontweight="bold",
                color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFFFFF", edgecolor=cor_eixo, linewidth=1.4, alpha=0.96),
                arrowprops=dict(arrowstyle="->", color=cor_eixo, linewidth=1.3),
                zorder=7
            )

        tot_mat = int(df["total_materias"].sum())
        tot_fat = len(df)
        ax.text(
            0.98,
            0.95,
            f"Amostra Factual Utilizada:\n{tot_mat:,.0f} notícias em {tot_fat:,.0f} fatos-ponte\n(Eixos: Político, Econômico e Social)".replace(",", "."),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0, alpha=0.96),
            zorder=10
        )

        ax.set_title("Régua Temporal de Fatos-Ponte e Divergência Editorial (2015–2025)",
                     fontsize=13.5, fontweight="bold", color="#0F172A", pad=15)
        ax.set_ylabel("Divergência de Sentimento Editorial [0 a 1]", fontsize=10, fontweight="bold", color="#334155")
        ax.set_xlabel("Linha do Tempo (Data de Eclosão do Evento Factual)", fontsize=10, fontweight="bold", color="#334155")
        ax.set_ylim(0.05, 0.49)
        ax.set_xlim(pd.to_datetime("2014-12-01"), pd.to_datetime("2025-12-31"))
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", fontsize=9.5)

        plt.subplots_adjust(top=0.91, bottom=0.18, left=0.06, right=0.96)
        plt.savefig(str(caminho_png), dpi=300)
        plt.close()
        return str(caminho_png)

    def gerar_todas_legendas(self) -> None:
        self.dir_saida_grafos_imagens.mkdir(parents=True, exist_ok=True)

        fig_pol, ax_pol = plt.subplots(figsize=(6.0, 1.25), dpi=300)
        fig_pol.patch.set_facecolor("#FFFFFF")
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        cmap = mcolors.LinearSegmentedColormap.from_list("pol_grad", [self.cor_tradicional, self.cores_eixos["Politico"], self.cor_digital])
        ax_pol.imshow(gradient, aspect="auto", cmap=cmap, extent=[-0.5, 0.5, 0, 1])
        ax_pol.set_xlim(-0.5, 0.5)
        ax_pol.set_ylim(0, 1)
        ax_pol.set_yticks([])
        ax_pol.set_xticks([-0.35, 0.0, 0.35])
        ax_pol.set_xticklabels(["Polo Tradicional\n(P < -0.15)", "Convergência\n(-0.15 a +0.15)", "Polo Digital\n(P > +0.15)"], fontsize=8, color="#334155")
        ax_pol.tick_params(axis="x", length=3, color="#94A3B8")
        for spine in ["top", "right", "left"]:
            ax_pol.spines[spine].set_visible(False)
        ax_pol.spines["bottom"].set_color("#CBD5E1")
        ax_pol.set_title("Polarização Editorial (Score P)", fontsize=9, fontweight="bold", pad=8, color="#1E293B")
        plt.tight_layout()
        plt.savefig(str(self.dir_saida_grafos_imagens / "01_legenda_grafo_planisferio_polarizacao.svg"), bbox_inches="tight")
        plt.savefig(str(self.dir_saida_grafos_imagens / "01_legenda_grafo_planisferio_polarizacao.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_pol)

        fig_sem, ax_sem = plt.subplots(figsize=(6.8, 1.35), dpi=300)
        ax_sem.axis("off")
        ax_sem.set_xlim(0, 1)
        ax_sem.set_ylim(0, 1)
        fig_sem.patch.set_facecolor("#FFFFFF")
        rect = patches.FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.02,rounding_size=0.04", edgecolor="#E2E8F0", facecolor="#F8FAFC", linewidth=1.0)
        ax_sem.add_patch(rect)
        ax_sem.text(0.04, 0.74, "Comunidades Temáticas de Palavras:", fontsize=8.5, fontweight="bold", color="#1E293B", va="center")
        eixos_lista = [
            (self.cores_eixos["Politico"], "Político", 0.04, 0.32),
            (self.cores_eixos["Economico"], "Econômico", 0.22, 0.32),
            (self.cores_eixos["Social"], "Social/Saúde", 0.44, 0.32),
            (self.cores_eixos["Ambiental"], "Ambiente", 0.68, 0.32),
            (self.cores_eixos["Internacional"], "Internacional", 0.04, 0.12)
        ]
        for cor, lbl, px, py in eixos_lista:
            ax_sem.plot(px, py, marker="s", markersize=6, color=cor)
            ax_sem.text(px + 0.025, py, lbl, fontsize=7.5, color="#334155", va="center")
        plt.tight_layout()
        plt.savefig(str(self.dir_saida_grafos_imagens / "02_legenda_grafo_semantico_palavras.svg"), bbox_inches="tight")
        plt.savefig(str(self.dir_saida_grafos_imagens / "02_legenda_grafo_semantico_palavras.png"), dpi=300, bbox_inches="tight")
        plt.close(fig_sem)

    def executar_todos(self) -> None:
        self.gerar_01_balanco_bipolar_sentimento()
        self.gerar_02_lollipop_clickbait()
        self.gerar_03a_cardiograma_ciclos_eleitorais()
        self.gerar_04_sankey_enquadramento()
        self.gerar_05a_radar_silhueta_editorial()
        self.gerar_05b_tornado_efeito_mudo()
        self.gerar_06a_serie_temporal_divergencia_tematica()
        self.gerar_06b_metastase_tematica_polarizacao()
        self.gerar_07_regua_betweenness()
        self.gerar_todas_legendas()
