# Quantificação Multidimensional e Modelagem Relacional da Cobertura Jornalística no Brasil (2015–2025)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/SQLite-Modo%20WAL-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="Banco SQLite" />
  <img src="https://img.shields.io/badge/spaCy-pt__core__news__lg-09A3D5?style=for-the-badge&logo=spacy&logoColor=white" alt="spaCy NER" />
  <img src="https://img.shields.io/badge/Transformers-BERTimbau%20(PT--BR)-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Transformers" />
</p>
<p align="center">
  <img src="https://img.shields.io/badge/HDBSCAN-Clusterização-4B0082?style=for-the-badge" alt="Clusterização HDBSCAN" />
  <img src="https://img.shields.io/badge/NetworkX-Análise%20de%20Redes-1F4E79?style=for-the-badge" alt="Modelagem em Grafos" />
  <img src="https://img.shields.io/badge/Plotly-Visualização%20Interativa-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Visualização Plotly" />
  <img src="https://img.shields.io/badge/SciPy-Testes%20Estatísticos-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" alt="Inferência Estatística" />
</p>

Este projeto consiste em uma infraestrutura computacional de ponta a ponta, modular e escalável para raspagem, extração, deduplicação criptográfica, vetorização semântica, agrupamento espaço-temporal de eventos factuais, extração de indicadores de enquadramento e *clickbait*, modelagem relacional em grafos e inferência estatística comparativa sobre a imprensa brasileira ao longo de uma década histórica (2015–2025).

O delineamento empírico estabelece uma comparação metodológica equilibrada entre dois ecossistemas editoriais: o **Grupo Tradicional Consolidado** (veículos originários da mídia impressa/broadcast de referência) e o **Grupo Digital Nativo** (veículos concebidos originariamente no ambiente web), totalizando dez veículos de abrangência nacional sob anonimização.

---

## Perguntas de Pesquisa e Hipóteses Formais

O pipeline foi projetado para responder quantitativamente a quatro eixos fundamentais da ciência da comunicação e do processamento de linguagem natural:

1. **Pergunta Principal (Divergência de Enquadramento no Mesmo Evento Factual):**  
   *Existe divergência sistemática e estatisticamente significativa na polaridade de sentimento e na atribuição de entidades entre veículos distintos ao noticiarem exatamente o mesmo evento factual?*  
   - **Metodologia:** Testes de variância ANOVA unidirecional e Kruskal-Wallis sobre a dispersão de sentimento e menções a entidades em agrupamentos densos (*clusters*) de eventos compartilhados.

2. **Pergunta Secundária 1 (Clickbait e Sensacionalismo Editorial):**  
   *Veículos digitais nativos recorrem com maior frequência e intensidade a estratégias de clickbait em títulos do que veículos tradicionais consolidados?*  
   - **Metodologia:** Teste paramétrico $t$ de Welch e teste não paramétrico Mann-Whitney U, acompanhados do cálculo do tamanho de efeito ($d$ de Cohen) sobre o escore multidimensional de *clickbait*.

3. **Pergunta Secundária 2 (Dinâmica Temporal em Ciclos Eleitorais):**  
   *A polarização e a intensidade de viés de enquadramento aumentam de forma estatisticamente significativa em anos de eleições gerais e municipais em comparação com períodos não eleitorais?*  
   - **Metodologia:** Análise de séries temporais desagregadas por ano e mês, contrastando anos eleitorais (2016, 2018, 2020, 2022, 2024) e não eleitorais via testes de hipótese bicaudais.

4. **Pergunta Secundária 3 (Efeito Mudo e Silenciamento Seletivo Temático):**  
   *Certos eixos temáticos (política, economia, internacional, direitos humanos, etc.) sofrem sub-representação desproporcional ou silenciamento em determinados veículos em comparação com a média do ecossistema?*  
   - **Metodologia:** Teste Qui-Quadrado ($\chi^2$) de independência em tabelas de contingência cruzando veículos e eixos temáticos canônicos.

---

## Arquitetura do Pipeline Metodológico

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. COLETA RESILIENTE (CDX Wayback Machine + Sitemaps Históricos)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 2. CASCATA ANTI-PAYWALL & POLÍTICA DE ACESSO ÉTICO                     │
│    (Snapshots Brutos / Googlebot JSON-LD / Headers Acadêmicos)        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 3. EXTRAÇÃO DE CONTEÚDO, LIMPEZA & DEDUPLICAÇÃO HASH (SHA-256)         │
│    (Trafilatura + Normalização Unicode NFKC + Descarte de Ruídos)      │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 4. PROCESSAMENTO DE LINGUAGEM NATURAL (NLP MULTIDIMENSIONAL)           │
│    - Embeddings Semânticos Densos (MiniLM-L12 / 384d)                  │
│    - Análise de Sentimento e Polaridade Neural (BERT em PT-BR)         │
│    - Detecção Heurística e Sintática de Clickbait                      │
│    - Reconhecimento de Entidades (NER spaCy pt_core_news_lg)           │
│    - Classificação em Eixos Temáticos Canônicos                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. AGRUPAMENTO ESPAÇO-TEMPORAL DE EVENTOS (HDBSCAN COM BLOCKING)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 6. MODELAGEM RELACIONAL EM GRAFOS DE SIMILARIDADE EDITORIAL            │
│    (Grafo Bipartido Veículos ↔ Eventos + Projeção Unipartida GEXF)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 7. INFERÊNCIA ESTATÍSTICA, RELATÓRIO CIENTÍFICO & GRÁFICOS PLOTLY      │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Componentes do Sistema

### 1. Coleta Histórica e Resiliência de Rede
- **Particionamento Temporal no CDX:** Consultas divididas ano a ano (2015 a 2025) para isolar falhas eventuais e maximizar a recuperação histórica.
- **Rate-Limiter Global Thread-Safe:** Mecanismo centralizado de bloqueio com delay de 2,5s garantindo espaçamento estrito entre requisições para conformidade com as APIs públicas do Internet Archive.
- **Paginação Eficiente com `showResumeKey`:** Ingestão de lotes leves de 1.000 URLs por requisição, evitando estouros de buffer e cancelamentos por timeout.
- **Gravação em Tempo Real (Streaming):** Inserção imediata em lotes no SQLite (`callback_lote`) à medida que as URLs chegam da rede.
- **Seleção Flexível de Fonte:** Opção CLI para coletar via CDX, Sitemaps diretos ou ambos simultaneamente.

### 2. Extração, Normalização e Deduplicação Criptográfica
- **Cascata Multi-Nível Anti-Paywall:** Estratégia de acesso em 4 camadas que combina snapshots estáticos no Wayback Machine, dados estruturados JSON-LD (`NewsArticle`), emulação Googlebot e renderização Trafilatura.
- **Limpeza de Ruídos e Boilerplate:** Remoção robusta de anúncios, cabeçalhos, rodapés, assinaturas de agências e caixas de comentários.
- **Normalização de Texto:** Normalização Unicode em forma NFKC, padronização de espaços e limpeza de caracteres de controle.
- **Deduplicação Determinística (SHA-256):** Geração de hash criptográfico único sobre $\text{SHA-256}(\text{título} + \text{corpo limpo})$, impedindo a inserção de matérias duplicadas.

### 3. Processamento de Linguagem Natural (NLP)
- **Vetorização Semântica:** Mapeamento de artigos em espaço vetorial denso contínuo (384 dimensões via `paraphrase-multilingual-MiniLM-L12-v2`).
- **Sentimento e Polaridade:** Classificação neural de polaridade (positivo, negativo, neutro) e pontuação contínua $[-1, +1]$ via BERT em português.
- **Métricas de Clickbait:** Algoritmo ponderado avaliando pontuação expressiva, proporção de maiúsculas, presença de números e gatilhos sensacionalistas.
- **Reconhecimento de Entidades (NER):** Identificação de pessoas, organizações e locais (via spaCy `pt_core_news_lg`) associada à análise de polaridade contextual.
- **Classificação Temática:** Atribuição supervisionada aos eixos canônicos: Político, Econômico, Social, Ambiental, Internacional e Esportivo.

### 4. Clusterização Espaço-Temporal de Eventos
- **Bloqueio Temporal (*Time-Window Blocking*):** Segmentação do corpus em janelas temporais deslizantes de 5 dias.
- **HDBSCAN Adaptativo:** Agrupamento por densidade espacial não paramétrico sobre a matriz de distâncias de cosseno, isolando ruídos (outliers $-1$).

### 5. Modelagem em Grafos de Redes Complexas
- **Grafo Bipartido $G_{bipartido}$:** Rede com dois conjuntos de nós (Veículos e Eventos Factuais), com arestas ponderadas por sentimento.
- **Projeção Unipartida $G_{veiculos}$:** Rede homogênea de veículos em que a força da ligação reflete concordância editorial e de enquadramento.
- **Exportação Interoperável:** Geração de arquivos GEXF para análise avançada no Gephi.

### 6. Análise Estatística e Relatórios
- **Testes Estatísticos Automatizados:** ANOVA, Kruskal-Wallis, teste $t$ de Welch, Mann-Whitney U, Qui-Quadrado de Efeito Mudo e correlação de Spearman.
- **Relatórios Standalone:** Compilação de relatórios analíticos em Markdown e dashboards interativos via Plotly.

---

## Configuração e Instalação

### Pré-requisitos
- Python 3.10 ou superior
- Gerenciador de pacotes `pip`
- Ambiente virtual Python configurado

### 1. Criar Ambiente Virtual e Ativar
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar Dependências e Modelo spaCy
```bash
pip install -r requirements.txt
python -m spacy download pt_core_news_lg
```

---

## Execução e Comandos da Linha de Comando (CLI)

O ponto de entrada principal do pipeline é o arquivo [`main.py`], que oferece execução modular por etapas ou em fluxo completo:

### 1. Coleta de URLs (`--etapa coletar`)

O módulo de coleta suporta filtros granulares por fonte, veículo específico e intervalo de anos para evitar consultas redundantes e focar exatamente nas lacunas necessárias:

```bash
# Coleta completa (Sitemaps dos portais + API CDX do Wayback Machine)
python main.py --etapa coletar --fonte todas

# Coleta focada exclusivamente no CDX (Acervo histórico 2015-2025 com rate-limiting seguro)
python main.py --etapa coletar --fonte cdx

# Coleta focada exclusivamente nos Sitemaps XML e RSS diretos dos portais
python main.py --etapa coletar --fonte sitemaps

# Coleta direcionada a um veículo específico (ex: veículo de interesse)
python main.py --etapa coletar --veiculo codigo_veiculo --fonte cdx

# Coleta filtrada por um único ano (ex: 2025)
python main.py --etapa coletar --ano 2025 --fonte cdx

# Coleta filtrada por intervalo de anos (ex: 2023 a 2025)
python main.py --etapa coletar --ano-inicio 2023 --ano-fim 2025 --fonte cdx

# Coleta combinada com limite por veículo (ideal para testes rápidos)
python main.py --etapa coletar --veiculo codigo_veiculo --ano 2024 --limite 100 --fonte cdx
```

### 2. Extração de Conteúdo e Deduplicação (`--etapa extrair`)

Executa o download das matérias, remoção de paywalls em cascata, limpeza com Trafilatura e deduplicação via SHA-256:

```bash
# Extração de todas as URLs pendentes no banco SQLite
python main.py --etapa extrair

# Extração com limite de matérias por veículo
python main.py --etapa extrair --limite 100
```

### 3. Processamento de Linguagem Natural (`--etapa nlp`)

Gera embeddings vetoriais, classifica sentimento neural, analisa menções a entidades (NER) e calcula scores de clickbait:

```bash
python main.py --etapa nlp
```

### 4. Clusterização Temporal de Eventos (`--etapa clusterizar`)

Agrupa matérias publicadas no mesmo intervalo temporal (janela de 5 dias) que cobrem o mesmo fato noticioso usando HDBSCAN:

```bash
python main.py --etapa clusterizar
```

### 5. Modelagem em Grafos (`--etapa grafos`)

Constrói o grafo bipartido Eventos-Veículos e a projeção de similaridade editorial, exportando redes em formato `.gexf` para o Gephi:

```bash
python main.py --etapa grafos
```

### 6. Inferência Estatística e Efeito Mudo (`--etapa estatisticas`)

Executa os testes de hipótese (ANOVA, Mann-Whitney, Qui-Quadrado de silenciamento temático e correlações de ciclos eleitorais):

```bash
python main.py --etapa estatisticas
```

### 7. Geração de Relatório e Dashboards Interativos (`--etapa relatorio`)

Gera o documento final de síntese em Markdown e relatórios HTML com gráficos interativos em Plotly:

```bash
python main.py --etapa relatorio
```

---

### 8. Execução Completa Sequencial (`--etapa tudo`)

Executa automaticamente todas as 7 etapas do pipeline em ordem lógica:

```bash
# Execução completa de ponta a ponta
python main.py --etapa tudo

# Execução completa com limite por veículo
python main.py --etapa tudo --limite 200
```

---

### 9. Comandos Utilitários de Diagnóstico e Banco de Dados

```bash
# Exibir painel com o status detalhado (separação CDX vs Sitemaps, matriz anual por jornal e eixos temáticos)
python main.py --status

# Limpar arquivos de logs de execuções anteriores
python main.py --limpar-logs

# Resetar completamente o banco de dados (recriar tabelas vazias)
python main.py --reset-db
```

---

## Conformidade Ética e Anonimização Metodológica

1. **Uso Exclusivamente Acadêmico:** A raspagem e a extração cumprem finalidade estrita de pesquisa científica, fundamentada no art. 46 da Lei de Direitos Autorais (Lei Federal nº 9.610/1998) e no Marco Civil da Internet (Lei Federal nº 12.965/2014).
2. **Boas Práticas de Coleta:** Respeito integral a diretivas `robots.txt`, cabeçalhos informativos identificando o projeto e taxas de requisição rigorosamente espaçadas para evitar sobrecarga em servidores.
3. **Anonimização na Apresentação Científica:** Na consolidação de relatórios e artigos, os veículos são identificados por designadores tipológicos anônimos (*Tradicional 1 a 5* e *Digital Nativo 1 a 5*) para preservar a isenção analítica e o foco na estrutura sistêmica da comunicação.
