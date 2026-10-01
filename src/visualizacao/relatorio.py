from pathlib import Path
from typing import Any, Dict
from src.configuracao import obter_configuracao
from src.registro import obter_logger
from src.banco import BancoDados


class GeradorRelatorio:
    def __init__(self, banco: BancoDados):
        self.banco = banco
        self.config = obter_configuracao()
        self.logger = obter_logger("visualizacao")
        self.dir_relatorios = Path(self.config.obter("geral.diretorio_relatorios", "relatorios"))
        self.dir_relatorios.mkdir(parents=True, exist_ok=True)

    def gerar_relatorio_markdown(
        self,
        estatisticas_banco: Dict[str, Any],
        respostas_perguntas: Dict[str, Any],
        metricas_grafos: Dict[str, Any]
    ) -> str:
        caminho_md = self.dir_relatorios / "relatorio_final.md"

        p1 = respostas_perguntas.get("pergunta_principal", {})
        p2 = respostas_perguntas.get("pergunta_clickbait", {})
        p3 = respostas_perguntas.get("pergunta_eleicoes", {})
        p4 = respostas_perguntas.get("pergunta_efeito_mudo", {})

        div_sent = p1.get("estatisticas_divergencia_sentimento", {})
        anova_p1 = p1.get("teste_efeito_veiculo_sentimento", {})

        cb_trad = p2.get("estatisticas_tradicionais", {})
        cb_dig = p2.get("estatisticas_digitais_nativos", {})
        teste_p2 = p2.get("teste_hipotese", {})

        el_eleit = p3.get("estatisticas_anos_eleitorais", {})
        el_nao_eleit = p3.get("estatisticas_anos_nao_eleitorais", {})
        teste_p3 = p3.get("teste_hipotese", {})

        mudo_analise = p4.get("analise_efeito_mudo", {})
        teste_p4 = p4.get("teste_independencia_qui_quadrado", {})

        conteudo = f"""# Quantificação Multidimensional e Modelagem Relacional da Cobertura Jornalística no Brasil (2015–2025)

**Autor:** Otávio Alves Lima  
**Disciplina:** Ciência dos Dados  

---

## 1. Sumário Executivo e Visão Geral
Este trabalho investigou de forma quantitativa e empírica o viés editorial, a divergência de enquadramento, o uso de recursos de sensacionalismo (*clickbait*) e o efeito mudo/silenciamento seletivo na cobertura jornalística brasileira entre 2015 e 2025, comparando dois blocos editoriais:
- **Grupo Tradicional (5 veículos):** Folha de S.Paulo, Estadão, O Globo, Gazeta do Povo e Correio Braziliense.
- **Grupo Digital Nativo (5 veículos):** Metrópoles, Poder360, O Antagonista, Brasil 247 e Nexo Jornal.

Para preservar a imparcialidade científica na apresentação dos dados, os veículos individuais são reportados de forma anonimizada (*Tradicional 1 a 5* e *Digital 1 a 5*), mantendo a agregação estrutural por Grupo Editorial.

### Indicadores Gerais da Base de Dados
- **Total de URLs Coletadas:** {estatisticas_banco.get('total_urls', 0):,}
- **Total de Matérias Validadas e Extraídas:** {estatisticas_banco.get('total_materias', 0):,}
- **Embeddings Vetoriais Gerados:** {estatisticas_banco.get('total_embeddings', 0):,}
- **Clusters de Eventos Factuais Pareados:** {estatisticas_banco.get('total_eventos', 0):,}
- **Nós no Grafo Bipartido (Veículos + Eventos):** {metricas_grafos.get('bipartido_nos', 0):,}
- **Arestas de Cobertura Bipartida:** {metricas_grafos.get('bipartido_arestas', 0):,}

---

## 2. Metodologia e Arquitetura do Pipeline

```
[CDX Web Archive (2015-2025) + Sitemaps/RSS Fallback]
                        │
                        ▼
           [Cascata Anti-Paywall e Robots.txt]
                        │
                        ▼
       [Trafilatura + Limpeza de Boilerplate NFKC]
                        │
                        ▼
        [Sentence-Transformers (Embeddings PT-BR)]
                        │
                        ▼
     [HDBSCAN + Blocking Temporal (Janela de 5 dias)]
                        │
                        ▼
 [Features de Viés: Sentimento (BERT-PT) + Clickbait + NER spaCy]
                        │
                        ▼
   [Grafo Bipartido Veículo-Evento & Projeção Unipartida]
                        │
                        ▼
 [Inferência Estatística (ANOVA, Mann-Whitney, Qui-Quadrado, Efeito Mudo)]
```

---

## 3. Resultados e Respostas às Perguntas de Pesquisa

### 3.1 Pergunta Principal: Divergência de Enquadramento no Mesmo Evento
> *Quando diferentes jornais cobrem o mesmo evento factual dentro de uma janela temporal curta, existe divergência sistemática e mensurável na polaridade de sentimento e enquadramento entre os grupos?*

- **Eventos com Cobertura Compartilhada (Multi-veículo):** {p1.get('total_eventos_compartilhados', 0)}
- **Divergência Média de Sentimento no Mesmo Evento (Desvio Padrão):** {div_sent.get('media', 0.0):.4f} (Mediana: {div_sent.get('mediana', 0.0):.4f})
- **Teste ANOVA (Efeito do Veículo no Sentimento):** F = {anova_p1.get('f_stat', 0.0):.4f}, p-valor = {anova_p1.get('f_pvalor', 1.0):.6e}
- **Teste Kruskal-Wallis:** H = {anova_p1.get('kruskal_h', 0.0):.4f}, p-valor = {anova_p1.get('kruskal_pvalor', 1.0):.6e}
- **Conclusão:** **{p1.get('conclusao', 'N/A')}**

---

### 3.2 Pergunta Secundária 1: Clickbait e Sensacionalismo (Tradicionais vs. Digitais Nativos)
> *Existe relação entre o uso de recursos de sensacionalismo textual (clickbait) e o tipo de veículo (tradicionais vs. digitais nativos)?*

- **Índice Médio de Clickbait no Grupo Tradicional:** {cb_trad.get('media', 0.0):.4f} ± {cb_trad.get('desvio_padrao', 0.0):.4f}
- **Índice Médio de Clickbait no Grupo Digital Nativo:** {cb_dig.get('media', 0.0):.4f} ± {cb_dig.get('desvio_padrao', 0.0):.4f}
- **Teste t de Welch:** t = {teste_p2.get('t_stat', 0.0):.4f}, p-valor = {teste_p2.get('t_pvalor', 1.0):.6e}
- **Teste Mann-Whitney U:** U = {teste_p2.get('mannwhitney_u', 0.0):.4f}, p-valor = {teste_p2.get('mannwhitney_pvalor', 1.0):.6e}
- **Tamanho de Efeito (Cohen's d):** {teste_p2.get('d_cohen', 0.0):.4f}
- **Conclusão:** **{p2.get('conclusao', 'N/A')}**

#### Distribuição de Clickbait por Veículo Anonimizado
| Posição | Veículo Anonimizado | Grupo Editorial | Média Clickbait | Desvio Padrão |
|:---:|:---|:---:|:---:|:---:|
"""
        ranking = p2.get("ranking_clickbait_veiculos", [])
        trad_i = 1
        dig_i = 1
        for pos, item in enumerate(ranking, start=1):
            if len(item) == 3:
                v_nome, v_tipo, stats_v = item
            else:
                v_nome, stats_v = item
                v_tipo = "-"
            if v_tipo == "tradicional":
                v_anon = f"Tradicional {trad_i}"
                trad_i += 1
                g_nome = "Grupo Tradicional"
            else:
                v_anon = f"Digital {dig_i}"
                dig_i += 1
                g_nome = "Grupo Digital Nativo"
            conteudo += f"| {pos} | {v_anon} | {g_nome} | {stats_v.get('media', 0.0):.4f} | {stats_v.get('desvio_padrao', 0.0):.4f} |\n"

        conteudo += f"""
---

### 3.3 Pergunta Secundária 2: Evolução Temporal em Ciclos Eleitorais
> *O grau de enviesamento textual aumenta de forma perceptível em anos de eleição (2016, 2018, 2020, 2022 e 2024) em comparação a anos sem eleição?*

- **Índice Composto de Viés em Anos Eleitorais:** {el_eleit.get('media', 0.0):.4f} ± {el_eleit.get('desvio_padrao', 0.0):.4f}
- **Índice Composto de Viés em Anos Não-Eleitorais:** {el_nao_eleit.get('media', 0.0):.4f} ± {el_nao_eleit.get('desvio_padrao', 0.0):.4f}
- **Teste t de Welch:** t = {teste_p3.get('t_stat', 0.0):.4f}, p-valor = {teste_p3.get('t_pvalor', 1.0):.6e}
- **Teste Mann-Whitney U:** U = {teste_p3.get('mannwhitney_u', 0.0):.4f}, p-valor = {teste_p3.get('mannwhitney_pvalor', 1.0):.6e}
- **Conclusão:** **{p3.get('conclusao', 'N/A')}**

---

### 3.4 Pergunta Secundária 3: Efeito Mudo e Silenciamento Seletivo por Eixo Temático
> *Existe assimetria na priorização de pautas e silenciamento relativo de eixos temáticos (político, econômico, social, ambiental, internacional, esportivo) entre a imprensa tradicional e os nativos digitais?*

- **Divergência Global de Efeito Mudo (Jensen-Shannon ponderado):** {mudo_analise.get('divergencia_efeito_mudo_global', 0.0):.4f}
- **Teste Qui-Quadrado de Independência de Pautas:** $\\chi^2$ = {teste_p4.get('chi2_stat', 0.0):.4f}, p-valor = {teste_p4.get('p_valor', 1.0):.6e} (Graus de Liberdade: {teste_p4.get('graus_liberdade', 0)})
- **Conclusão:** **{p4.get('conclusao', 'N/A')}**

#### Detalhamento por Área Temática
| Eixo Temático | Proporção Tradicional | Proporção Digital | Razão Trad/Dig | Índice Efeito Mudo |
|:---|:---:|:---:|:---:|:---:|
"""
        eixos_dict = mudo_analise.get("eixos", {})
        for eixo_nome, e_info in eixos_dict.items():
            conteudo += f"| {eixo_nome.capitalize()} | {e_info.get('proporcao_tradicional', 0.0)*100:.1f}% | {e_info.get('proporcao_digital', 0.0)*100:.1f}% | {e_info.get('razao_cobertura_trad_dig', 0.0):.2f}x | {e_info.get('indice_efeito_mudo', 0.0):.4f} |\n"

        conteudo += f"""
---

## 4. Modelagem em Grafo e Similaridade Editorial

Foram construídos dois modelos de rede anonimizados:
1. **Grafo Bipartido $G_{{bipartido}}$ (Veículos $\\leftrightarrow$ Eventos):** Permite verificar a sobreposição de pauta factual entre os 10 veículos dos dois grupos. Exportado em formato GEXF para Gephi em `relatorios/grafos/rede_bipartida.gexf`.
2. **Projeção Unipartida de Veículos $G_{{veiculos}}$:** Arestas ponderadas pela proximidade de enquadramento nos mesmos eventos factuais. Exportado para Gephi em `relatorios/grafos/rede_veiculos.gexf` e interativo em `relatorios/graficos/rede_similaridade_interativa.html`.

---

## 5. Arquivos e Gráficos Interativos Disponíveis
- [Gráfico Interativo de Sentimento (Anonimizado)](graficos/distribuicao_sentimento_veiculos.html)
- [Gráfico Interativo de Clickbait por Grupo](graficos/clickbait_por_tipo_veiculo.html)
- [Evolução Temporal dos Ciclos Eleitorais](graficos/evolucao_temporal_vies_eleicoes.html)
- [Efeito Mudo e Silenciamento Seletivo por Eixo Temático](graficos/efeito_mudo_por_area_tematica.html)
- [Radar Multidimensional Editorial por Grupo](graficos/radar_enquadramento_editorial.html)
- [Rede de Proximidade Editorial Anonimizada](graficos/rede_similaridade_interativa.html)
- [Arquivo GEXF para Gephi (Rede Bipartida)](grafos/rede_bipartida.gexf)
- [Arquivo GEXF para Gephi (Projeção Veículos)](grafos/rede_veiculos.gexf)

---
*Relatório gerado automaticamente pelo pipeline de pesquisa.*
"""
        with open(caminho_md, "w", encoding="utf-8") as f:
            f.write(conteudo)

        self._gerar_versao_html(conteudo)
        self.logger.info(f"Relatorios gerados com sucesso em {caminho_md}")
        return str(caminho_md)

    def _gerar_versao_html(self, markdown_texto: str) -> str:
        caminho_html = self.dir_relatorios / "relatorio_final.html"
        linhas = markdown_texto.split("\n")
        html_body = []

        for l in linhas:
            if l.startswith("# "):
                html_body.append(f"<h1>{l[2:]}</h1>")
            elif l.startswith("## "):
                html_body.append(f"<h2>{l[3:]}</h2>")
            elif l.startswith("### "):
                html_body.append(f"<h3>{l[4:]}</h3>")
            elif l.startswith("#### "):
                html_body.append(f"<h4>{l[5:]}</h4>")
            elif l.startswith("- "):
                html_body.append(f"<li>{l[2:]}</li>")
            elif l.startswith("> "):
                html_body.append(f"<blockquote>{l[2:]}</blockquote>")
            elif l.strip() == "---":
                html_body.append("<hr/>")
            elif l.strip():
                html_body.append(f"<p>{l}</p>")

        corpo = "\n".join(html_body)
        documento = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Relatório Final de Pesquisa - Cobertura Jornalística</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #fafafa; }}
h1, h2, h3 {{ color: #111; }}
h1 {{ border-bottom: 2px solid #eaeaea; padding-bottom: 12px; }}
h2 {{ border-bottom: 1px solid #eaeaea; padding-bottom: 8px; margin-top: 30px; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; background: #fff; }}
th, td {{ border: 1px solid #ddd; padding: 10px; text-align: left; }}
th {{ background-color: #f2f2f2; }}
blockquote {{ border-left: 4px solid #1f77b4; padding-left: 15px; color: #555; background: #f0f7fb; margin: 15px 0; padding: 10px 15px; }}
hr {{ border: 0; height: 1px; background: #eee; margin: 30px 0; }}
li {{ margin-bottom: 6px; }}
</style>
</head>
<body>
{corpo}
</body>
</html>
"""
        with open(caminho_html, "w", encoding="utf-8") as f:
            f.write(documento)
        return str(caminho_html)
