# Quantificação de Viés Editorial e Sensacionalismo na Mídia Brasileira (2015–2025): Uma Abordagem Baseada em Processamento de Linguagem Natural e Modelagem em Grafos

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/Base-3.03M%20Matérias-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="Base de 3 Milhões de Matérias" />
  <img src="https://img.shields.io/badge/Eventos-446.649%20Clusters-4CAF50?style=for-the-badge" alt="446k Eventos Pareados" />
  <img src="https://img.shields.io/badge/spaCy-pt__core__news__lg-09A3D5?style=for-the-badge&logo=spacy&logoColor=white" alt="spaCy NER" />
  <img src="https://img.shields.io/badge/Transformers-MiniLM%20%7C%20RoBERTa-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Transformers" />
  <img src="https://img.shields.io/badge/Gephi-GEXF%20%7C%20Redes%20Complexas-1F4E79?style=for-the-badge" alt="Modelagem em Grafos" />
  <img src="https://img.shields.io/badge/Figuras-PNG%20300%20DPI%20%7C%20SVG-8CAAE6?style=for-the-badge" alt="Figuras em Alta Resolução" />
  <img src="https://img.shields.io/badge/Open%20Data-Zenodo%20FAIR-blue?style=for-the-badge" alt="Zenodo Open Science" />
</p>

---

## Sumário

1. [Visão Geral do Projeto](#1-visão-geral-do-projeto)
2. [Indicadores Gerais da Base de Dados](#2-indicadores-gerais-da-base-de-dados)
3. [Perguntas de Pesquisa e Conclusões Científicas](#3-perguntas-de-pesquisa-e-conclusões-científicas)
   - [3.1 Pergunta Principal: Divergência de Enquadramento no Mesmo Evento Factual](#31-pergunta-principal-divergência-de-enquadramento-no-mesmo-evento-factual)
   - [3.2 Pergunta Secundária 1: Sensacionalismo Textual e Clickbait](#32-pergunta-secundária-1-sensacionalismo-textual-e-clickbait)
   - [3.3 Pergunta Secundária 2: Dinâmica Temporal e Ciclos Eleitorais](#33-pergunta-secundária-2-dinâmica-temporal-e-ciclos-eleitorais)
   - [3.4 Pergunta Secundária 3: Efeito Mudo e Silenciamento Seletivo Temático](#34-pergunta-secundária-3-efeito-mudo-e-silenciamento-seletivo-temático)
   - [3.5 Genealogia da Polarização: Quando e Como Começou no Brasil](#35-genealogia-da-polarização-quando-e-como-começou-no-brasil)
4. [Pipeline de Produção de Cada Etapa](#4-pipeline-de-produção-de-cada-etapa)
   - [Etapa 1: Coleta de URLs e Rastreamento Histórico](#etapa-1-coleta-de-urls-e-rastreamento-histórico)
   - [Etapa 2: Extração Textual, Limpeza de Conteúdo e Bypass Anti-Paywall](#etapa-2-extração-textual-limpeza-de-conteúdo-e-bypass-anti-paywall)
   - [Etapa 3: Processamento de Linguagem Natural Multidimensional (NLP)](#etapa-3-processamento-de-linguagem-natural-multidimensional-nlp)
   - [Etapa 4: Agrupamento Espaço-Temporal de Eventos Factuais (Clustering)](#etapa-4-agrupamento-espaço-temporal-de-eventos-factuais-clustering)
   - [Etapa 5: Modelagem de Redes Complexas e Grafos de Conhecimento](#etapa-5-modelagem-de-redes-complexas-e-grafos-de-conhecimento)
   - [Etapa 6: Inferência Estatística e Testes de Hipótese](#etapa-6-inferência-estatística-e-testes-de-hipótese)
   - [Etapa 7: Visualização Científica para o Artigo (PNG 300 DPI e SVG)](#etapa-7-visualização-científica-para-o-artigo-png-300-dpi-e-svg)
   - [Etapa 8: Empacotamento de Ciência Aberta e Depósito no Zenodo](#etapa-8-empacotamento-de-ciência-aberta-e-depósito-no-zenodo)
5. [Figuras e Gráficos de Alto Impacto para o Artigo (`resultados/graficos/`)](#5-figuras-e-gráficos-de-alto-impacto-para-o-artigo-resultadosgraficos)
6. [Modelos de Redes Complexas e Imagens (`resultados/`)](#6-modelos-de-redes-complexas-e-imagens-resultados)
   - [6.1 Catálogo dos 4 Modelos de Grafos](#61-catálogo-dos-4-modelos-de-grafos)
   - [6.2 Renderizações dos Grafos (PNG e SVG)](#62-renderizações-dos-grafos-png-e-svg)
7. [Arquitetura de Pastas do Repositório](#7-arquitetura-de-pastas-do-repositório)
8. [Repositórios de Ciência Aberta e Zenodo](#8-repositórios-de-ciência-aberta-e-zenodo)
9. [Como Instalar o Ambiente](#9-como-instalar-o-ambiente)
10. [Equipe de Pesquisa e Autoria](#10-equipe-de-pesquisa-e-autoria)

---

## 1. Visão Geral do Projeto

Este projeto consiste em uma infraestrutura computacional científica de ponta a ponta para raspagem, normalização textual, deduplicação criptográfica, vetorização semântica (*Sentence-Transformers*), agrupamento espaço-temporal de eventos factuais (*HDBSCAN* com janela deslizante de 5 dias), quantificação multidimensional de sentimento e *clickbait*, modelagem relacional em grafos complexos e inferência estatística comparativa sobre a cobertura jornalística brasileira entre **janeiro de 2015 e dezembro de 2025**.

O desenho experimental estabelece um confronto quantitativo e empírico entre dois blocos editoriais:
- **Grupo Tradicional Consolidado (5 veículos):** Veículos centenários de referência da mídia impressa/broadcast nacional (*Folha de S.Paulo, Estadão, O Globo, Gazeta do Povo e Correio Braziliense*).
- **Grupo Digital Nativo (5 veículos):** Portais de grande audiência concebidos nativamente para a web (*Metrópoles, Poder360, O Antagonista, Brasil 247 e Nexo Jornal*).

Para garantir estrita conformidade com os princípios éticos da pesquisa científica e reprodutibilidade aberta, os veículos individuais são reportados de forma rigorosamente anonimizada sob as convenções **`Tradicional 1 a 5`** e **`Digital 1 a 5`**, preservando a integridade das inferências ao nível de grupo e de ator institucional.

---

## 2. Indicadores Gerais da Base de Dados

| Indicador Estrutural | Total Quantificado | Descrição Metodológica |
| :--- | :---: | :--- |
| **Total de URLs Coletadas** | **4.720.199** | Extraídas de índices CDX da *Wayback Machine* e sitemaps históricos |
| **Total de Matérias Validadas** | **3.034.033** | Textos íntegros com limpeza de *boilerplate* e validação SHA-256 |
| **Embeddings Vetoriais Gerados** | **3.034.033** | Vetores densos de 384 dimensões (*paraphrase-multilingual-MiniLM-L12-v2*) |
| **Clusters de Eventos Factuais** | **446.649** | Macro-eventos pareados via HDBSCAN e janelas deslizantes de 5 dias |
| **Eventos com Cobertura Compartilhada** | **254.737** | Fatos cobertos simultaneamente por múltiplos veículos concorrentes |
| **Nós nos Grafos Bipartidos** | **20.010** | 10 veículos de imprensa + 20.000 macro-eventos balanceados em 6 eixos |
| **Arestas de Cobertura Ponderada** | **82.227** | Conexões ponderadas por volume de matérias e distâncias geodésicas |
| **Período Histórico Coberto** | **10 Anos** | De 01/01/2015 a 31/12/2025 (cobrindo 5 ciclos eleitorais nacionais) |

---

## 3. Perguntas de Pesquisa e Conclusões Científicas

### 3.1 Pergunta Principal: Divergência de Enquadramento no Mesmo Evento Factual
> *Quando diferentes jornais cobrem o mesmo evento factual dentro de uma janela temporal curta, existe divergência sistemática e mensurável na polaridade de sentimento e enquadramento entre os grupos?*

- **Eventos Analisados com Cobertura Compartilhada:** 254.737 macro-eventos pareados.
- **Divergência Média de Sentimento no Mesmo Fato (Desvio Padrão):** 0,1956 (Mediana: 0,1926).
- **Teste ANOVA One-Way (Efeito do Veículo no Sentimento):** $F = 134.545,95$, $p < 0,000001$.
- **Teste Kruskal-Wallis:** $H = 878.220,57$, $p < 0,000001$.
- **Conclusão Científica:** **Existe divergência sistemática e estatisticamente significativa no enquadramento e sentimento**. A linha editorial e o modelo de negócios do veículo condicionam fortemente a valência emocional impressa sobre o mesmo acontecimento da realidade factual.

### 3.2 Pergunta Secundária 1: Sensacionalismo Textual e Clickbait
> *Existe relação entre o uso de recursos de sensacionalismo textual (clickbait) e o tipo de veículo (imprensa tradicional vs. nativos digitais)?*

- **Índice Médio de Clickbait no Grupo Tradicional:** 0,0626 ± 0,0712
- **Índice Médio de Clickbait no Grupo Digital Nativo:** 0,0588 ± 0,0705
- **Média Geral Nacional:** 0,0607
- **Teste $t$ de Welch:** $t = -47,0483$, $p < 0,000001$
- **Teste Mann-Whitney U:** $U = 1.105.198.441.631,00$, $p < 0,000001$
- **Tamanho de Efeito ($d$ de Cohen):** **-0,0540 (Classificado como Desprezível)**
- **Conclusão Científica:** **Não há diferença substantiva nos níveis de sensacionalismo entre os modelos**. A hipótese de senso comum de que os portais digitais nativos operam com níveis muito superiores de sensacionalismo é **estatisticamente refutada**. As técnicas apelativas de manchete foram assimiladas transversalmente por ambos os ecossistemas como resposta comercial às regras de engajamento algorítmico das plataformas digitais.

### 3.3 Pergunta Secundária 2: Dinâmica Temporal e Ciclos Eleitorais
> *Como os ciclos eleitorais brasileiros impactam a polarização de sentimento, o volume de produção e a divergência editorial ao longo do tempo?*

- **Média de Sentimento em Anos Eleitorais:** -0,0198 ± 0,2140
- **Média de Sentimento em Anos Não-Eleitorais:** -0,0085 ± 0,2105
- **Teste de Hipótese Bicaudal:** $p < 0,000001$
- **Conclusão Científica:** A cobertura jornalística brasileira possui uma **pulsação bienal sincronizada aos ciclos eleitorais** (2016, 2018, 2020, 2022, 2024). Nos meses de campanha (agosto a outubro dos anos pares), o sentimento editorial atinge seus vales mais negativos, e a divergência editorial entre os blocos atinge ápices históricos (ultrapassando 0,195 em 2018 e 2022).

### 3.4 Pergunta Secundária 3: Efeito Mudo e Silenciamento Seletivo Temático
> *Existe evidência empírica de 'efeito mudo' ou silenciamento seletivo de pautas temáticas entre os grupos editoriais?*

- **Proporção da Cobertura de Economia:** Tradicionais = 19,4% vs. Nativos Digitais = 3,2% (**Queda de -16,2 pontos percentuais; Razão de 6,06 para 1**).
- **Proporção da Cobertura de Política:** Tradicionais = 67,7% vs. Nativos Digitais = 85,9% (**Aumento de +18,2 pontos percentuais**).
- **Teste Qui-Quadrado de Independência:** $\chi^2 = 1.306.906,00$, $p < 0,000001$ ($V$ de Cramér = 0,2075).
- **Conclusão Científica:** **Comprova-se a hipótese do Efeito Mudo**. Os veículos digitais nativos silenciam quase que integralmente a cobertura macroeconômica analítica (taxa Selic, balança comercial, política fiscal), canalizando mais de 85% de sua capacidade produtiva para disputas e bastidores da arena política.

### 3.5 Genealogia da Polarização: Quando e Como Começou no Brasil
- **Marco Zero (Quando Começou):** **Agosto de 2016** (votação final do impeachment e posse de Michel Temer). O índice de divergência atingiu o pico histórico absoluto da década (**0,2221**). A partir desse momento, a divergência jamais retornou aos patamares basais pré-2016 (0,15–0,16).
- **Mecanismo de Propagação (Como Começou - Metástase Temática Pós-2018):** Entre 2015 e 2017, a polarização estava restrita quase exclusivamente à arena partidária. A partir de 2018 e da pandemia em 2020, ocorreu uma "metástase temático-editorial":
  - **Meio Ambiente:** Salto de **+28,8%** na divergência editorial (de 0,1413 para 0,1820).
  - **Saúde e Sociedade:** Salto de **+21,0%** na divergência editorial (de 0,1482 para 0,1793).
  - Fatos científicos, vacinas e desmatamento foram colonizados pelo atrito ideológico.

---

## 4. Pipeline de Produção de Cada Etapa

A infraestrutura foi construída como um pipeline modular de 8 estágios reprodutíveis:

```
[Etapa 1: Coleta CDX/Sitemaps] ──> [Etapa 2: Extração & Anti-Paywall] ──> [Etapa 3: NLP Multidimensional]
                                                                                     │
[Etapa 6: Inferência Estatística] <── [Etapa 5: Grafos & Gephi] <── [Etapa 4: Clustering de Eventos]
              │
              ├──> [Etapa 7: Visualização Artigo (PNG 300 DPI e SVG)]
              └──> [Etapa 8: Pacotes FAIR Zenodo (CSVs, ZIPs & Metadados)]
```

### Detalhamento dos Estágios de Produção:

#### Etapa 1: Coleta de URLs e Rastreamento Histórico
- **Objetivo:** Identificar e catalogar os links históricos publicados pelos 10 veículos entre 2015 e 2025.
- **Métodos & Tecnologias:** Consultas indexadas à API CDX do *Internet Archive / Wayback Machine* e raspagem concorrente de sitemaps XML históricos (`urllib3`, `BeautifulSoup`).
- **Resiliência:** Detecção dinâmica de throttling HTTP 429/503, espera exponencial com *jitter*, rotação de user-agents e persistência transacional com status `pendente`.
- **Comando de Execução:**
  ```bash
  python3 main.py --etapa coletar --ano-inicio 2015 --ano-fim 2025
  # Opcional: filtrar por veículo ou limite específico
  python3 main.py --etapa coletar --veiculo tradicional_1 --limite 50000
  ```

#### Etapa 2: Extração Textual, Limpeza de Conteúdo e Bypass Anti-Paywall
- **Objetivo:** Efetuar o download do conteúdo integral das matérias, extrair metadados estruturados e contornar barreiras de paywall.
- **Métodos & Tecnologias:** Cascata anti-paywall baseada em múltiplos seletores heurísticos (`trafilatura`, `newspaper3k`, `lxml`), emulação de referrers de redes sociais e renderização de cabeçalhos de arquivo.
- **Higienização:** Remoção sistemática de blocos de scripts, tags de rastreamento, anúncios, menus de navegação e publicidade nativa. Validação e deduplicação via cálculo de *hash* criptográfico SHA-256 do corpo textual.
- **Comando de Execução:**
  ```bash
  python3 main.py --etapa extrair
  # Reprocessamento de eventuais falhas de download
  python3 main.py --etapa extrair --reprocessar-erros
  ```

#### Etapa 3: Processamento de Linguagem Natural Multidimensional (NLP)
- **Objetivo:** Enriquecer cada matéria jornalística com representações vetoriais densas, métricas de valência emocional, índice de sensacionalismo e entidades nomeadas.
- **Métodos & Modelos:**
  1. *Embeddings Semânticos:* Modelo `paraphrase-multilingual-MiniLM-L12-v2` quantizado em ONNX (vetores densos de 384 dimensões por matéria).
  2. *Análise de Sentimento:* Modelo RoBERTa multilíngue (`pysentimiento`) quantizado em ONNX para cálculo de probabilidades de sentimento positivo, negativo e neutro, consolidado em score contínuo $[-1.0, +1.0]$.
  3. *Sensacionalismo / Clickbait:* Modelo híbrido estatístico-léxico que mensura pontuação expressiva (`!?`), pronomes deícticos na manchete, termos de urgência cognitiva, superlativos e razão de caracteres em caixa alta.
  4. *Extração de Entidades (NER):* Pipeline `spacy` com modelo `pt_core_news_lg` identificando e vinculando pessoas, organizações e locais.
- **Comando de Execução:**
  ```bash
  python3 main.py --etapa nlp
  # Ou execução seletiva por sub-módulo:
  python3 main.py --etapa embeddings
  python3 main.py --etapa sentimento
  python3 main.py --etapa entidades
  ```

#### Etapa 4: Agrupamento Espaço-Temporal de Eventos Factuais (Clustering)
- **Objetivo:** Identificar o mesmo acontecimento factual da realidade noticiado por múltiplos veículos concorrentes.
- **Métodos & Algoritmos:** Agrupador temporal com janela móvel deslizante de 5 dias (calibrada pelo decaimento do ciclo de atenção jornalística). Aplicação do algoritmo `HDBSCAN` sobre a matriz de distância de cosseno dos embeddings com refinamento TF-IDF para geração de títulos sintéticos dos macro-eventos.
- **Comando de Execução:**
  ```bash
  python3 main.py --etapa clusterizar
  ```

#### Etapa 5: Modelagem de Redes Complexas e Grafos de Conhecimento
- **Objetivo:** Modelar as relações estruturais da imprensa nacional sob a ótica de redes bipartidas, projeções unipartidas e topologia semântica.
- **Modelos Construídos:**
  - *Modelo 1 (Cascata Hierárquica Multinível):* 20.010 nós e 82.227 arestas conectando veículos a arquipélagos temáticos da realidade.
  - *Modelo 2 (Planisfério Cósmico de Polarização):* 20.010 nós espacializados por gravitação de polarização ForceAtlas2 contínua.
  - *Modelo 3 (Grafo de Similaridade Editorial dos Veículos):* Rede projetada de 10 veículos com similaridade de cosseno e categorização intra/inter-blocos.
  - *Modelo 4 (Grafo Semântico de Palavras e Conceitos):* Rede de coocorrência conceitual textual sem bolinhas (tipografia ponderada por PageRank e comunidades Louvain).
- **Comando de Execução:**
  ```bash
  python3 main.py --etapa grafos
  ```

#### Etapa 6: Inferência Estatística e Testes de Hipótese
- **Objetivo:** Executar testes estatísticos formais que respondem a todas as perguntas de pesquisa da investigação.
- **Métodos Computacionais:** Teste ANOVA One-Way, teste Kruskal-Wallis, teste $t$ de Welch, teste Mann-Whitney $U$, teste Qui-Quadrado de Independência ($\chi^2$), $d$ de Cohen, $V$ de Cramér e análises de caminhos geodésicos em grafos via Dijkstra e BFS.
- **Comando de Execução:**
  ```bash
  python3 main.py --etapa estatisticas
  # Ou gerar o relatório consolidado em texto:
  python3 main.py --etapa relatorio
  ```

#### Etapa 7: Visualização Científica para o Artigo (PNG 300 DPI e SVG)
- **Objetivo:** Gerar todas as 10 figuras científicas do manuscrito e as renderizações dos grafos (PNG e SVG).
- **Produtos Gerados:**
  - 10 Figuras em PNG 300 DPI (`resultados/graficos/`).
  - Grafos base e prontos em subpastas `png/` e `svg/` (`resultados/grafos_base/` e `resultados/grafos_prontos/`).
- **Comando de Execução:**
  ```bash
  python3 src/visualizacao/gerador_graficos_artigo.py
  ```

#### Etapa 8: Empacotamento de Ciência Aberta e Depósito no Zenodo
- **Objetivo:** Preparar os pacotes padronizados de replicação em conformidade com as diretrizes FAIR e esquemas de metadados DataCite / OpenAIRE.
- **Produtos Gerados:** `dados_graficos_zenodo.zip` e `dados_grafos_zenodo.zip`, acompanhados de seus respectivos `zenodo_metadata.json` e codebooks metodológicos em `README_ZENODO.md`.
- **Comando de Execução:**
  ```bash
  python3 base_dados_limpa/gerar_zenodo.py
  ```

---

## 5. Figuras e Gráficos de Alto Impacto para o Artigo (`resultados/graficos/`)

Todas as figuras foram produzidas em **PNG de altíssima resolução (300 DPI)** com diagramação profissional para manuscritos científicos (LaTeX, Overleaf, Word):

| Figura | Arquivo PNG (300 DPI) | Tipo de Visualização | Pergunta Respondida no Artigo |
| :---: | :--- | :--- | :--- |
| **Fig. 1** | [`01_balanco_bipolar_sentimento.png`](resultados/graficos/01_balanco_bipolar_sentimento.png) | **Balanço Bipolar de Densidade** | Distribuição e assimetria da densidade de sentimento entre os 10 veículos. |
| **Fig. 2** | [`02_lollipop_ranking_clickbait.png`](resultados/graficos/02_lollipop_ranking_clickbait.png) | **Lollipop Chart com Média Nacional** | Ranking de sensacionalismo com média (0,0607) e P90 extremo. |
| **Fig. 3A** | [`03a_cardiograma_ciclos_eleitorais.png`](resultados/graficos/03a_cardiograma_ciclos_eleitorais.png) | **Cardiograma de Pulsação Eleitoral** | Série temporal decenal evidenciando vales emocionais nos anos de eleição. |
| **Fig. 3B** | [`03b_matriz_calendario_polarizacao.png`](resultados/graficos/03b_matriz_calendario_polarizacao.png) | **Matriz Calendário Mês a Mês** | Mapa de calor decenal confirmando picos sistemáticos de atrito em outubro. |
| **Fig. 4** | [`04_sankey_enquadramento_editorial.png`](resultados/graficos/04_sankey_enquadramento_editorial.png) | **Fluxo Sankey Multicamada** | Trajetória do enquadramento: Grupo $\rightarrow$ Eixo Temático $\rightarrow$ Sentimento. |
| **Fig. 5A** | [`05a_radar_silhueta_editorial.png`](resultados/graficos/05a_radar_silhueta_editorial.png) | **Radar Polar de Silhueta Editorial** | Contraste de especialização e cobertura temática entre tradicional e digital. |
| **Fig. 5B** | [`05b_tornado_efeito_mudo.png`](resultados/graficos/05b_tornado_efeito_mudo.png) | **Gráfico Tornado do Efeito Mudo** | Prova do silenciamento econômico (-16,2 p.p.) e hipertrofia política (+18,2 p.p.). |
| **Fig. 6A** | [`06a_serie_temporal_divergencia_tematica.png`](resultados/graficos/06a_serie_temporal_divergencia_tematica.png) | **Série Histórica com Marco Zero** | Estopim em Agosto/2016 (pico 0,2221) e liderança da arena política. |
| **Fig. 6B** | [`06b_metastase_tematica_polarizacao.png`](resultados/graficos/06b_metastase_tematica_polarizacao.png) | **Metástase Temática da Polarização** | Contágio da polarização para Meio Ambiente (+28,8%) e Saúde (+21,0%) pós-2018. |
| **Fig. 7** | [`07_regua_polaridade_betweenness.png`](resultados/graficos/07_regua_polaridade_betweenness.png) | **Régua Bipolar de Fatos-Ponte** | Dispersão dos fatos com raio calibrado por Betweenness Centrality. |

---

## 6. Modelos de Redes Complexas e Imagens (`resultados/`)

### 6.1 Catálogo dos 4 Modelos de Grafos
1. **Modelo 1 — Cascata Hierárquica Multinível (`grafo_cascata_hierarquica.gexf`):**
   - 20.010 nós e 82.227 arestas ponderadas.
   - Fluxo bipartido: veículos no topo, 6 arquipélagos temáticos da realidade na base.
   - Cores temáticas padronizadas: Político (`#8B5CF6`), Econômico (`#EAB308`), Social (`#EC4899`), Meio Ambiente (`#10B981`), Internacional (`#06B6D4`), Esportivo (`#E11D48`).
2. **Modelo 2 — Planisfério Cósmico de Polarização (`grafo_planisferio_polarizacao.gexf`):**
   - 20.010 nós em coordenadas bidimensionais contínuas ForceAtlas2.
   - Posição governada pelo índice gravitacional de polarização $P \in [-1.0, +1.0]$.
   - Gradiente de três zonas: Polo Tradicional (`#1E3A8A`), Centro de Convergência (`#8B5CF6`) e Polo Digital (`#EA580C`).
3. **Modelo 3 — Grafo de Similaridade Editorial entre Veículos (`grafo_similaridade_veiculos.gexf`):**
   - 10 veículos de comunicação conectados por 45 arestas ponderadas por similaridade de cosseno ($0,1718$ a $0,7784$).
   - Arestas categorizadas em Intra-Tradicional (azul), Intra-Digital (laranja) e Inter-Blocos (roxa tracejada).
4. **Modelo 4 — Grafo Semântico de Palavras e Conceitos (`grafo_semantico_palavras.gexf`):**
   - 310 termos-chave jornalísticos com tipografia direta ponderada por PageRank (sem círculos/bolinhas nos nós).
   - Cores por comunidades de modularidade Louvain associadas aos grandes eixos discursivos.

### 6.2 Renderizações dos Grafos (PNG e SVG)
- **Grafos Base:**
  - PNGs em `resultados/grafos_base/grafos_imagens/png/`
  - SVGs em `resultados/grafos_base/grafos_imagens/svg/`
- **Grafos Prontos para o Artigo:**
  - PNGs 300 DPI em `resultados/grafos_prontos/grafos_imagens/png/`
  - SVGs vetoriais em `resultados/grafos_prontos/grafos_imagens/svg/`

---

## 7. Arquitetura de Pastas do Repositório

```text
web-scrap/
├── config.yaml                              # Configurações globais centralizadas
├── main.py                                  # Ponto de entrada CLI do pipeline de produção
├── base_dados_global/                       # Armazenamento bruto e banco central
│   ├── jornalismo.db                        # Banco SQLite WAL (23 GB, 3M matérias, 446k eventos)
│   └── modelos/                             # Modelos quantizados ONNX (RoBERTa / MiniLM)
├── base_dados_limpa/                        # Dados higienizados para replicação e Ciência Aberta
│   ├── dados_graficos/                      # Dados usados na geração das figuras
│   │   ├── csv/                             # Tabelas CSV tabuladas e documentadas
│   │   └── zenodo/                          # Pacote Zenodo (dados_graficos_zenodo.zip, metadata, codebook)
│   └── dados_grafos/                        # Dados usados na modelagem de redes complexas
│       ├── csv/                             # Listas de nós, arestas e matrizes em CSV
│       └── zenodo/                          # Pacote Zenodo (dados_grafos_zenodo.zip, metadata, codebook)
├── resultados/                              # Produtos e entregas de visualização
│   ├── graficos/                            # 10 Figuras PNG em 300 DPI prontas para o artigo
│   ├── grafos_base/                         # Modelos de rede base
│   │   ├── grafos_arquivo/                  # Arquivos .gexf e .gephi (Cascata, Planisfério, Semântico, Similaridade)
│   │   └── grafos_imagens/                  # Renderizações dos grafos base
│   │       ├── png/                         # Imagens em formato PNG
│   │       └── svg/                         # Imagens vetoriais em formato SVG
│   └── grafos_prontos/                      # Modelos de rede prontos para defesa e publicação
│       ├── grafos_arquivo/                  # Arquivos .gexf prontos
│       └── grafos_imagens/                  # Figuras renderizadas para o artigo
│           ├── png/                         # Imagens PNG em alta resolução (300 DPI)
│           └── svg/                         # Imagens vetoriais infinitas (SVG)
├── projeto final/                           # Espaço reservado para o manuscrito final do artigo
├── ideias/                                  # Cadernos metodológicos e tutoriais analíticos
│   ├── 01_o_que_os_grafos_representam_e_uso_no_gephi.txt
│   ├── 02_justificativa_clusterizacao_janela_5_dias.txt
│   ├── 03_regua_de_ano_polaridade_e_betweenness.txt
│   ├── 04_quando_e_como_comecou_a_polarizacao_no_brasil.txt
│   ├── 05_guia_de_analise_e_interpretacao_dos_graficos_no_artigo.txt
│   ├── 06_guia_de_analise_e_interpretacao_dos_grafos_no_artigo.txt
│   ├── 07_tutorial_gephi_grafo_de_palavras_sem_bolinhas.txt
│   └── 08_tutorial_gephi_grafo_similaridade_veiculos.txt
└── src/                                     # Código-fonte modular (Zero comentários '#')
    ├── banco.py                             # Camada de persistência SQLite WAL
    ├── configuracao.py                      # Carregador de configurações
    ├── registro.py                          # Sistema estruturado de logs
    ├── coleta/                              # Motores de rastreamento de URLs
    ├── extracao/                            # Motores de raspagem e anti-paywall
    ├── nlp/                                 # Embeddings, sentimento e clickbait
    ├── cluster/                             # Clusterização temporal HDBSCAN
    ├── analise/                             # Testes estatísticos e inferência
    ├── grafos/                              # Construtores de redes e grafos GEXF
    └── visualizacao/                        # Geradores de figuras, grafos e relatórios
```

---

## 8. Repositórios de Ciência Aberta e Zenodo

Para assegurar a auditabilidade, reprodutibilidade e conformidade com os princípios FAIR (*Findable, Accessible, Interoperable, Reusable*), os dados do projeto foram estruturados em pacotes autônomos prontos para depósito no **Zenodo**:

1. **Pacote de Redes Complexas (`base_dados_limpa/dados_grafos/zenodo/`):**
   - Arquivo: `dados_grafos_zenodo.zip` contendo listas completas de nós, arestas e matrizes de adjacência.
   - Metadados: `zenodo_metadata.json` em padrão OpenAIRE / DataCite.
   - Documentação: `README_ZENODO.md` com codebook detalhado de cada coluna e dicionário de variáveis.
2. **Pacote de Séries Temporais e Indicadores (`base_dados_limpa/dados_graficos/zenodo/`):**
   - Arquivo: `dados_graficos_zenodo.zip` contendo todas as séries históricas, distribuições e matrizes.
   - Metadados: `zenodo_metadata.json` com resumo metodológico.
   - Documentação: `README_ZENODO.md` com codebook das variáveis analíticas.

---

## 9. Como Instalar o Ambiente

### 9.1 Pré-requisitos
- Linux (Ubuntu 22.04+ ou Fedora 38+)
- Python 3.10 ou superior
- SQLite 3.35+ (com suporte a modo WAL e funções matemáticas)

### 9.2 Instalação das Dependências
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib kaleido
```

---

## 10. Equipe de Pesquisa e Autoria

- **Otávio Alves Lima** — UFCI-MG (Pesquisador Principal)
- **Pedro Francisco Sousa Silva** — UFCI-MG (Pesquisador Principal)
