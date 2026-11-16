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
4. [Instalação do Ambiente e Execução via Interface Gráfica](#4-instalação-do-ambiente-e-execução-via-interface-gráfica)
   - [4.1 Pré-requisitos do Sistema](#41-pré-requisitos-do-sistema)
   - [4.2 Instalação das Dependências](#42-instalação-das-dependências)
   - [4.3 Inicialização da Aplicação Desktop](#43-inicialização-da-aplicação-desktop)
   - [4.4 Navegação e Ordem Metodológica do Pipeline](#44-navegação-e-ordem-metodológica-do-pipeline)
   - [4.5 Recursos Operacionais da Interface](#45-recursos-operacionais-da-interface)
5. [Figuras e Gráficos de Alto Impacto para o Artigo (`resultados/graficos/`)](#5-figuras-e-gráficos-de-alto-impacto-para-o-artigo-resultadosgraficos)
6. [Modelos de Redes Complexas e Imagens (`resultados/grafos/`)](#6-modelos-de-redes-complexas-e-imagens-resultadosgrafos)
   - [6.1 Catálogo dos 2 Modelos Oficiais de Grafos](#61-catálogo-dos-2-modelos-oficiais-de-grafos)
   - [6.2 Estrutura Unificada de Arquivos e Imagens](#62-estrutura-unificada-de-arquivos-e-imagens)
7. [Arquitetura de Pastas do Repositório (Apenas Pastas Versionadas)](#7-arquitetura-de-pastas-do-repositório-apenas-pastas-versionadas)
8. [Repositórios de Ciência Aberta e Replicação](#8-repositórios-de-ciência-aberta-e-replicação)
9. [Equipe de Pesquisa e Autoria](#9-equipe-de-pesquisa-e-autoria)

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

## 4. Instalação do Ambiente e Execução via Interface Gráfica

Para executar a infraestrutura e reproduzir as análises, primeiro configure o ambiente virtual com as dependências do projeto e, em seguida, inicialize a aplicação desktop integrada.

### 4.1 Pré-requisitos do Sistema
- **Sistema Operacional:** Linux (Fedora 38+, Ubuntu 22.04+ ou distribuição equivalente).
- **Python:** Versão 3.10 ou superior.
- **SQLite:** Versão 3.35+ (com suporte nativo ao modo WAL e funções matemáticas).

---

### 4.2 Instalação das Dependências

Clone o repositório, crie um ambiente virtual isolado e instale os pacotes determinísticos de produção:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 4.3 Inicialização da Aplicação Desktop

Com o ambiente virtual ativado, inicialize a aplicação desktop em Flet:

```bash
python main.py
```

*(Alternativamente, `python main.py --gui` executa a mesma rotina de inicialização).*

A interface foi projetada em **tema estritamente claro (Light Mode)**, com design limpo, tipografia profissional e paleta de cores corporativa baseada no design system Material 3. Não é mais necessário executar scripts fragmentados por linha de comando nem memorizar parâmetros de terminal para cada estágio: o usuário opera todo o pipeline científico interativamente dentro da aplicação.

---

### 4.4 Navegação e Ordem Metodológica do Pipeline

O seletor principal de etapas organiza o fluxo de trabalho na ordem lógica e cronológica da metodologia científica:

```text
[1. Coleta de URLs] ──> [2. Extração & Paywall] ──> [3. Processamento NLP] ──> [4. Clustering de Eventos]
                                                                                     │
[7. Figuras do Artigo] <── [6. Construção de Grafos] <── [5. Inferência Estatística] <┘
                                       │
                    [8. Executar Tudo (Pipeline Completo)]
```

1. **1. Coleta de URLs e Rastreamento Histórico:**
   - Descoberta e catalogação de matérias jornalísticas publicadas entre 2015 e 2025.
   - *Parâmetros na Interface:* Limite de matérias por veículo, intervalo de anos (ex: `2015-2025` ou anos avulsos `2018, 2022`) e seleção de fonte (*Wayback Machine*, *Sitemaps XML* ou *Todas*).
2. **2. Extração de Matérias e Bypass Anti-Paywall:**
   - Download assíncrono do corpo textual integral, limpeza heurística (scripts, anúncios, menus) e validação de integridade por *hash* criptográfico SHA-256.
3. **3. Processamento de Linguagem Natural (NLP):**
   - Enriquecimento multidimensional das matérias:
     - *Embeddings Semânticos:* Modelo `paraphrase-multilingual-MiniLM-L12-v2` (384 dimensões).
     - *Sentimento Editorial:* Modelo RoBERTa multilíngue quantizado em ONNX (score contínuo $[-1.0, +1.0]$).
     - *Sensacionalismo / Clickbait:* Modelo híbrido que quantifica pontuação expressiva (`!?`), deíticos, urgência cognitiva e taxa de caixa alta.
4. **4. Agrupamento Espaço-Temporal de Eventos (Clustering):**
   - Pareamento do mesmo fato factual cobrindo veículos concorrentes via algoritmo `HDBSCAN` com janela móvel deslizante de 5 dias calibrada pelo ciclo de atenção jornalística.
5. **5. Análise Estatística e Perguntas de Pesquisa:**
   - Computação formal da bateria de testes de hipótese (ANOVA One-Way, Kruskal-Wallis, teste $t$ de Welch, Mann-Whitney $U$, Qui-Quadrado $\chi^2$ e Razão de Segregação Geodésica).
   - Renderização dos resultados diretamente na tela em cards visuais com ícones, métricas numéricas, $p$-valores e conclusões acadêmicas.
6. **6. Construção de Grafos para o Gephi (GEXF):**
   - Construção dos dois modelos oficiais de redes complexas: *Planisfério Cósmico da Polarização* (20.010 nós) e *Grafo Semântico de Palavras* (310 termos).
   - Cálculo automático e gravação nos arquivos GEXF dos algoritmos de detecção de comunidades Louvain ($Q$), centralidade de intermediação (Betweenness / Fatos-Ponte), caminhos geodésicos de Dijkstra e busca em largura (BFS).
7. **7. Geração de Figuras do Artigo (300 DPI):**
   - Renderização das 10 figuras científicas de alto impacto do manuscrito e legendas oficiais dos grafos em PNG 300 DPI e SVG vetorial.
   - O menu secundário permite escolher gerar todas as figuras de uma vez ou qualquer figura individual para inspeção rápida.
8. **8. Executar Tudo (Pipeline Completo):**
   - Execução sequencial automatizada de todas as etapas do início ao fim, ideal para replicação integral ou reprocessamento completo.

---

### 4.5 Recursos Operacionais da Interface

- **Execução Multithread Responsiva:** Tarefas pesadas rodam em threads de segundo plano sem bloquear ou congelar a interface. Indicadores visuais de status informam em tempo real o tempo de execução e o estado da tarefa (*Pronto*, *Em execução*, *Concluído com sucesso* ou *Erro*).
- **Console Integrado de Logs com Filtro por Etapa:**
  - O terminal não é poluído; todas as mensagens e eventos de logging são direcionados para o painel de logs da interface.
  - O usuário pode utilizar o dropdown de filtro de logs para visualizar apenas as mensagens de uma etapa específica (ex: somente logs de `grafos` ou `estatisticas`) ou a visão consolidada (`todas`).
- **Suporte Nativo a Dados Externos (Ciência Aberta / Replicação):**
  - A caixa de seleção *"Usar dados externos (para replicação científica)"* ativa um seletor de diretório (`FilePicker`) para apontar qualquer pasta local contendo os arquivos CSV ou pacotes ZIP, permitindo gerar figuras e grafos sem necessidade do banco central de 23 GB.
- **Acesso Direto aos Arquivos Gerados:**
  - Botões integrados na área de resultados permitem abrir instantaneamente as pastas no gerenciador de arquivos do sistema operacional:
    - `Abrir Pasta de Gráficos (PNG)` &rarr; abre `resultados/graficos/`.
    - `Abrir Pasta de Grafos (Arquivos e Imagens)` &rarr; abre `resultados/grafos/`.

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

## 6. Modelos de Redes Complexas e Imagens (`resultados/grafos/`)

### 6.1 Catálogo dos 2 Modelos Oficiais de Grafos
1. **Modelo 1 — Planisfério Cósmico da Polarização (`grafo_planisferio_polarizacao.gexf`):**
   - 20.010 nós em coordenadas bidimensionais contínuas ForceAtlas2.
   - Posição governada pelo índice gravitacional de polarização $P \in [-1.0, +1.0]$.
   - Gradiente de três zonas: Polo Tradicional (`#1E3A8A`), Centro de Convergência (`#8B5CF6`) e Polo Digital (`#EA580C`).
2. **Modelo 2 — Grafo Semântico de Palavras e Coocorrência (`grafo_semantico_palavras.gexf`):**
   - 310 termos-chave jornalísticos com tipografia direta ponderada por PageRank e Betweenness (sem círculos/bolinhas nos nós).
   - Cores por comunidades de modularidade Louvain associadas aos grandes eixos discursivos ($Q = 0,4018$).

### 6.2 Renderizações e Legendas dos Grafos (PNG e SVG)
- **Estrutura Unificada (`resultados/grafos/`):**
  - **Modelos e Dados (`resultados/grafos/arquivos/`):** Arquivos `.gexf` e `.gephi` para importação e manipulação no Gephi.
  - **Imagens e Legendas (`resultados/grafos/imagens/`):** Todas as renderizações oficiais dos grafos (PNG e SVG) e suas respectivas legendas (`01_legenda_grafo_planisferio_polarizacao.*`, `02_legenda_grafo_semantico_palavras.*`), centralizadas sem subpastas adicionais.

---

## 7. Arquitetura de Pastas do Repositório (Apenas Pastas Versionadas)

Abaixo está a estrutura estrita de arquivos e pastas que compõem o repositório de código e que serão efetivamente versionadas no Git quando o autor ordenar (excluindo os dados brutos de 23 GB, logs voláteis, ambientes virtuais e pastas locais presentes no `.gitignore`):

```text
web-scrap/
├── .gitignore                               # Regras de exclusão do Git (banco de dados 23GB, logs, venv, rascunhos)
├── config.yaml                              # Configurações globais centralizadas, limiares estatísticos e parâmetros
├── main.py                                  # Ponto de entrada do sistema que inicializa a aplicação desktop
├── requirements.txt                         # Especificação estrita de dependências Python para o ambiente
├── README.md                                # Documentação técnica completa, metodologia e guia de operação
├── modelos/                                 # Modelos ONNX e pesos quantizados para inferência local offline
├── resultados/                              # Produtos e entregas científicas de visualização e replicação
│   ├── graficos/                            # 10 Figuras científicas em PNG 300 DPI prontas para o manuscrito
│   ├── dados_graficos/                      # Tabelas CSV tabuladas para replicação analítica direta
│   └── grafos/                              # Estrutura unificada de redes complexas
│       ├── arquivos/                        # Modelos GEXF para Gephi e projeto mestre nativo 'grafos.gephi'
│       └── imagens/                         # Renderizações em alta resolução (PNG e SVG) e legendas oficiais
└── src/                                     # Código-fonte modular do sistema (Estritamente zero comentários '#')
    ├── __init__.py                          # Inicializador do pacote raiz
    ├── anti_paywall.py                      # Heurísticas de contorno a barreiras de assinatura e paywalls dinâmicos
    ├── banco.py                             # Camada de persistência relacional SQLite em modo WAL (thread-safe)
    ├── configuracao.py                      # Carregador e validador singleton de configurações YAML
    ├── interface.py                         # Interface gráfica desktop em Flet (tema claro, multithread, logs e cards)
    ├── registro.py                          # Sistema centralizado de logs com despachante thread-safe para a interface
    ├── robos.py                             # Utilitários de rede, rotação de headers e emulação de navegadores
    ├── analise/                             # Módulo de inferência e testes estatísticos
    │   ├── __init__.py                      # Inicializador do subpacote de análise
    │   ├── estatistica.py                   # Funções matemáticas de suporte (ANOVA, Kruskal-Wallis, Mann-Whitney)
    │   └── perguntas.py                     # Respondedor formal das 4 perguntas centrais do artigo acadêmico
    ├── cluster/                             # Módulo de agrupamento espaço-temporal de eventos
    │   ├── __init__.py                      # Inicializador do subpacote de clusterização
    │   └── agrupador.py                     # Motor HDBSCAN com janela móvel de 5 dias sobre embeddings semânticos
    ├── coleta/                              # Módulo de rastreamento e descoberta de URLs
    │   ├── __init__.py                      # Inicializador do subpacote de coleta
    │   ├── cdx.py                           # Cliente de consulta à API CDX do Wayback Machine (Internet Archive)
    │   ├── filtros.py                       # Normalização canônica, sanitização e descarte de URLs estáticas
    │   ├── motor_coleta.py                  # Orquestrador resiliente de coleta histórica por veículo e ano
    │   └── sitemaps.py                      # Parser concorrente de sitemaps XML jornalísticos históricos
    ├── extracao/                            # Módulo de raspagem textual e higienização
    │   ├── __init__.py                      # Inicializador do subpacote de extração
    │   ├── extrator.py                      # Extratores heurísticos de corpo textual e metadados de notícia
    │   ├── limpador.py                      # Remoção de anúncios, publicidade nativa e scripts de rastreamento
    │   └── motor_extracao.py                # Pipeline de download com validação por hash criptográfico SHA-256
    ├── grafos/                              # Módulo de modelagem de redes complexas
    │   ├── __init__.py                      # Inicializador do subpacote de grafos
    │   └── construtor_grafo.py              # Construtor GEXF do Planisfério e Semântico (Louvain, BFS, Dijkstra, Betweenness)
    ├── nlp/                                 # Módulo de processamento de linguagem natural
    │   ├── __init__.py                      # Inicializador do subpacote de NLP
    │   ├── atribuicao.py                    # Extração de autoria e agências noticiosas
    │   ├── classificador_tematico.py        # Classificação discursiva nos 5 grandes eixos temáticos do estudo
    │   ├── clickbait.py                     # Detecção híbrida estatística-léxica de sensacionalismo em manchetes
    │   ├── embeddings.py                    # Vetorização semântica densa (Sentence-Transformers MiniLM 384d)
    │   ├── entidades.py                     # Reconhecimento de entidades nomeadas (NER via spaCy)
    │   ├── motor_nlp.py                     # Orquestrador em lote de pipelines de NLP
    │   └── sentimento.py                    # Inferência de polaridade emocional via RoBERTa multilíngue
    └── visualizacao/                        # Módulo de visualização científica
        ├── __init__.py                      # Inicializador do subpacote de visualização
        └── graficos.py                      # Renderização profissional das 10 figuras em 300 DPI e legendas dos grafos
```

### Detalhamento das Responsabilidades de Cada Arquivo:

| Arquivo / Módulo | Responsabilidade Técnica no Pipeline |
| :--- | :--- |
| `main.py` | Ponto de entrada que inicializa a aplicação desktop gráfica em Flet (`InterfaceFletApp`). |
| `config.yaml` | Centraliza credenciais, URLs base, thresholds de similaridade de embeddings e esquemas de cores. |
| `requirements.txt` | Lista determinística de bibliotecas Python necessárias para execução do pipeline. |
| `.gitignore` | Protege o repositório contra arquivos pesados (excluindo `banco_dados/base.db` de 23 GB, `projeto_final/` e `.venv/`). |
| `src/interface.py` | Controla toda a interface visual desktop: seleção das 8 etapas, logs dinâmicos filtrados por etapa, cards de teste de hipótese e botões de sistema. |
| `src/banco.py` | Camada SQLite relacional operando em modo WAL com conexão thread-safe, índices analíticos e funções de salvamento estatístico. |
| `src/registro.py` | Despacha mensagens de log em tempo real tanto para a UI do Flet quanto para os manipuladores internos. |
| `src/anti_paywall.py` | Implementa seletores de contorno a muros de pagamento, headers de redes sociais e fallback de arquivamento. |
| `src/robos.py` | Gerencia pool de conexões HTTP, rotação de User-Agents e regras de atraso com jitter contra bloqueios. |
| `src/coleta/filtros.py` | Valida padrões regex de links de notícia e descarta arquivos de imagem, vídeo ou páginas estáticas. |
| `src/coleta/motor_coleta.py` | Executa o rastreamento concorrente na API CDX do Wayback Machine e nos sitemaps históricos. |
| `src/extracao/motor_extracao.py` | Realiza download, higienização do texto e gera a assinatura SHA-256 para evitar matérias duplicadas. |
| `src/nlp/classificador_tematico.py` | Classifica matérias nos 5 macro-eixos discursivos (Político, Econômico, Social, Ambiental, Internacional). |
| `src/nlp/sentimento.py` | Calcula valência emocional contínua $[-1, +1]$ e pontuação combinada de sensacionalismo/clickbait. |
| `src/cluster/agrupador.py` | Aplica HDBSCAN com janela móvel de 5 dias para identificar matérias de veículos concorrentes cobrindo o mesmo fato. |
| `src/analise/perguntas.py` | Executa ANOVA, Kruskal-Wallis, Mann-Whitney U, Qui-Quadrado e Razão de Segregação Geodésica. |
| `src/grafos/construtor_grafo.py` | Gera os arquivos GEXF do Planisfério da Polarização e Grafo Semântico, calculando Louvain, BFS, Dijkstra e Betweenness. |
| `src/visualizacao/graficos.py` | Produz as 10 figuras em alta resolução (300 DPI) para o artigo e renderiza as legendas oficiais em PNG e SVG. |
| `resultados/graficos/` | Contém as figuras finais em PNG 300 DPI prontas para publicação (Figuras 01, 02, 03A, 04, 05A, 05B, 06A, 06B, 07). |
| `resultados/grafos/arquivos/` | Contém os arquivos de rede `grafo_planisferio_polarizacao.gexf`, `grafo_semantico_palavras.gexf` e o projeto `grafos.gephi`. |
| `resultados/grafos/imagens/` | Reúne as renderizações gráficas dos grafos e todas as legendas oficiais em PNG e SVG. |
| `resultados/dados_graficos/` | Contém as tabelas CSV tabuladas para replicação analítica independente sem necessidade do banco SQLite. |

---

## 8. Repositórios de Ciência Aberta e Replicação

Para assegurar a total auditabilidade e reprodutibilidade científica sem exigir a transferência do banco central SQLite de 23 GB, o repositório disponibiliza diretamente os dados processados e normalizados em:

1. **Séries Temporais e Matrizes Estatísticas (`resultados/dados_graficos/`):**
   - Tabelas CSV contendo todas as variáveis agregadas, distribuições de densidade de sentimento, índices de clickbait e séries temporais mensais decenais (2015–2025).
2. **Modelos de Redes Complexas (`resultados/grafos/arquivos/`):**
   - Modelos GEXF e projeto Gephi contendo as matrizes de adjacência, coordenadas gravitacionais ForceAtlas2 e classes de modularidade de Louvain já computadas.

---

## 9. Equipe de Pesquisa e Autoria

- **Otávio Alves Lima** — UFCI-MG (Pesquisador Principal)
- **Pedro Francisco Sousa Silva** — UFCI-MG (Pesquisador Principal)
