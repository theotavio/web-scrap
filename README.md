# Quantificação Multidimensional e Modelagem Relacional da Cobertura Jornalística no Brasil (2015–2025)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/SQLite-Modo%20WAL-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="Banco SQLite" />
  <img src="https://img.shields.io/badge/Streaming-Micro--lotes%20(Baixa%20RAM)-4CAF50?style=for-the-badge" alt="Processamento em Streaming" />
  <img src="https://img.shields.io/badge/spaCy-pt__core__news__lg-09A3D5?style=for-the-badge&logo=spacy&logoColor=white" alt="spaCy NER" />
  <img src="https://img.shields.io/badge/Transformers-MiniLM%20%7C%20RoBERTa-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face Transformers" />
</p>
<p align="center">
  <img src="https://img.shields.io/badge/HDBSCAN-Clusterização%20(5%20dias)-4B0082?style=for-the-badge" alt="Clusterização HDBSCAN" />
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
│ 4. PROCESSAMENTO TEXTUAL & NLP MODULAR (STREAMING EM MICRO-LOTES)      │
│    ├─ Embeddings Vetoriais (MiniLM-L12 / 384d) + Eixos com Guardas     │
│    ├─ Sentimento Neural (pysentimiento RoBERTa) & Detecção Clickbait   │
│    └─ Reconhecimento de Entidades (NER spaCy) & Atribuição de Contexto │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ 5. AGRUPAMENTO ESPAÇO-TEMPORAL DE EVENTOS (JANELA DE 5 DIAS / HDBSCAN) │
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

### 3. Processamento de Linguagem Natural (NLP) Modular & Eficiente
- **Aceleração com ONNX Runtime INT8 & PyTorch Nativo:** O motor de sentimento utiliza inferência direta em tensores com quantização dinâmica em inteiros de 8 bits (INT8) via ONNX Runtime, contornando o *overhead* de wrappers e reduzindo o tempo de inferência em CPU em mais de 6x, com fallback automático para PyTorch (`torch.inference_mode()`).
- **Atribuição Contextual de Entidades Otimizada em Lote & Foco em Atores Centrais:** O cálculo de sentimento ao redor de menções a entidades (NER) foca as janelas neurais RoBERTa estritamente nas menções aos atores políticos e institucionais principais (`PER` e `ORG` - top 3 entidades e 1 janela principal), com inferência vetorial agrupada em lote no ONNX INT8 e índices parciais no SQLite (`idx_materias_ner_pendente` e `idx_materias_evento_ner`). Entidades geográficas (`LOC`) e secundárias herdam a polaridade global da matéria, acelerando o processamento em mais de 7x.
- **Processamento em Streaming Contínuo:** Todas as consultas operam em micro-lotes paginados (`LIMIT 500` para embeddings e `LIMIT 150-250` para features linguísticas), eliminando estouros de memória RAM e permitindo processar 3+ milhões de artigos sem sobrecarregar o sistema.
- **Carregamento Sob Demanda (*Lazy Loading*) & Descarte Dinâmico:** Os pesos das redes neurais (`SentenceTransformer`, `pysentimiento` e `spaCy`) são instanciados apenas no momento do cálculo e descarregados da memória (`gc.collect()`) entre as fases, mantendo o consumo de RAM estritamente abaixo de **1,5 GB a 2 GB**.
- **Classificação Temática com Precedência Editorial:** A editoria humana contida na URL original (`/politica/`, `/economia/`, `/esportes/`, `/mundo/`, etc.) tem prioridade máxima. O classificador semântico atua como *fallback* apenas em URLs genéricas.
- **Guardas Semânticas contra Metáforas:** Fórmulas de proteção impedem que metáforas esportivas corriqueiras na crônica política brasileira (*"queda de braço"*, *"arena"*, *"juiz de fora"*, *"time de ministros"*, *"disputa de poder"*) classifiquem notícias de governo e agentes públicos erroneamente como esporte.
- **Subetapas Textuais Desacopladas:** Possibilidade de executar de forma independente: apenas Embeddings, apenas Sentimento/Clickbait ou apenas Entidades/NER.
- **Pausa Graciosa (`Ctrl + C`):** Qualquer etapa pode ser interrompida a qualquer momento sem corrupção de banco; o progresso concluído é gravado imediatamente e a execução pode ser retomada de onde parou.

### 4. Clusterização Espaço-Temporal de Eventos
- **Bloqueio Temporal (*Time-Window Blocking*):** Segmentação do corpus em janelas deslizantes de **5 dias** (ciclo clássico de vida de notícias factuais). Impede o agrupamento indevido de notícias de momentos diferentes e viabiliza a comparação equilibrada entre mídia digital (imediata) e tradicional (reflexiva).
- **HDBSCAN Adaptativo:** Agrupamento por densidade espacial não paramétrico sobre a matriz de distâncias de cosseno, isolando ruídos (outliers $-1$).

### 5. Modelagem em Grafos de Redes Complexas
- **Grafo Bipartido $G_{bipartido}$:** Rede com dois conjuntos de nós (Veículos e Eventos Factuais), com arestas ponderadas por sentimento e volume de cobertura.
- **Projeção Unipartida $G_{veiculos}$:** Rede homogênea de veículos em que a força da ligação reflete concordância editorial e enquadramento conjunto sobre os mesmos acontecimentos.
- **Exportação Interoperável:** Geração de arquivos GEXF para análise avançada de modularidade e centralidade no Gephi.

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

O módulo de coleta suporta filtros granulares por fonte, veículo específico e intervalo de anos para evitar consultas redundantes:

```bash
# Coleta completa (Sitemaps dos portais + API CDX do Wayback Machine)
python main.py --etapa coletar --fonte todas

# Coleta focada exclusivamente no CDX (Acervo histórico 2015-2025 com rate-limiting seguro)
python main.py --etapa coletar --fonte cdx

# Coleta focada exclusivamente nos Sitemaps XML e RSS diretos dos portais
python main.py --etapa coletar --fonte sitemaps

# Coleta direcionada a um veículo específico
python main.py --etapa coletar --veiculo metropoles --fonte cdx

# Coleta filtrada por um único ano (ex: 2025)
python main.py --etapa coletar --ano 2025 --fonte cdx

# Coleta filtrada por intervalo de anos (ex: 2023 a 2025)
python main.py --etapa coletar --ano-inicio 2023 --ano-fim 2025 --fonte cdx

# Coleta combinada com limite por veículo (ideal para testes rápidos)
python main.py --etapa coletar --veiculo poder360 --ano 2024 --limite 100 --fonte cdx
```

---

### 2. Extração de Conteúdo e Deduplicação (`--etapa extrair`)

Executa o download das matérias, remoção de paywalls em cascata, limpeza com Trafilatura e deduplicação via SHA-256:

```bash
# Extração de todas as URLs pendentes no banco SQLite
python main.py --etapa extrair

# Extração com limite de matérias (para testes ou controle térmico)
python main.py --etapa extrair --limite 100

# Extração filtrada por veículo específico
python main.py --etapa extrair --veiculo folha --limite 500

# Reprocessar apenas URLs que falharam anteriormente com erro_download
python main.py --etapa extrair --reprocessar-erros

# Reprocessar erros de um veículo específico com limite
python main.py --etapa extrair --veiculo estadao --reprocessar-erros --limite 100
```

---

### 3. Processamento Textual & NLP (Modular e Desacoplado)

Para evitar sobrecarga de processamento, consumo excessivo de memória e aquecimento de hardware, a etapa de NLP foi particionada em **subetapas independentes**:

```bash
# -------------------------------------------------------------
# SUBETAPA A: Apenas Embeddings Vetoriais e Eixos Temáticos
# (Ultrarrápida, usa MiniLM-L12, não aquece a máquina)
# -------------------------------------------------------------
python main.py --etapa embeddings

# Embeddings com limite controlado ou veículo específico
python main.py --etapa embeddings --limite 1000 --veiculo metropoles

# -------------------------------------------------------------
# SUBETAPA B: Apenas Sentimento Neural (RoBERTa) & Clickbait
# (Calcula polaridade contínua e probabilidade neutra sem carregar spaCy)
# -------------------------------------------------------------
python main.py --etapa sentimento

# Sentimento em lote controlado
python main.py --etapa sentimento --limite 500 --veiculo folha

# -------------------------------------------------------------
# SUBETAPA C: Apenas Extração de Entidades Nomeadas (NER spaCy)
# (Mapeamento de agentes políticos, instituições e contexto de atribuição)
# -------------------------------------------------------------
python main.py --etapa entidades

# Processar entidades apenas para matérias pertencentes a eventos factuais agrupados
# (Recomendado após executar --etapa clusterizar; foca estritamente nos clusters do corpus analítico)
python main.py --etapa entidades --apenas-eventos

# Entidades em lote controlado
python main.py --etapa entidades --limite 500 --veiculo g1

# -------------------------------------------------------------
# FLUXO COMPLETO DE NLP (Executa A, B e C sequencialmente)
# (Descarrega os modelos da RAM entre cada fase)
# -------------------------------------------------------------
python main.py --etapa nlp --limite 1000
```

> [!TIP]
> **Pausa sem perdas:** Você pode interromper qualquer subetapa de NLP a qualquer momento pressionando **`Ctrl + C`**. Cada micro-lote concluído já foi salvo em disco no SQLite, e ao reexecutar o comando o sistema retoma exatamente da matéria seguinte.

---

### 4. Clusterização Temporal de Eventos (`--etapa clusterizar`)

Agrupa matérias publicadas no mesmo intervalo temporal (janela deslizante de 5 dias) que cobrem o mesmo fato noticioso usando HDBSCAN sobre os vetores de embeddings:

```bash
python main.py --etapa clusterizar
```

---

### 5. Modelagem em Grafos (`--etapa grafos`)

Constrói o grafo bipartido Eventos-Veículos e a projeção unipartida de similaridade editorial, exportando redes em formato `.gexf` para visualização e análise no Gephi:

```bash
python main.py --etapa grafos
```

---

### 6. Inferência Estatística e Efeito Mudo (`--etapa estatisticas`)

Executa os testes de hipótese formais (ANOVA, Kruskal-Wallis, Mann-Whitney, Qui-Quadrado de silenciamento temático e correlações temporais de ciclos eleitorais):

```bash
python main.py --etapa estatisticas
```

---

### 7. Geração de Relatório e Dashboards Interativos (`--etapa relatorio`)

Gera o documento final de síntese acadêmica em Markdown e relatórios HTML com gráficos interativos via Plotly:

```bash
python main.py --etapa relatorio
```

---

### 8. Execução Completa Sequencial (`--etapa tudo`)

Executa automaticamente todas as etapas do pipeline em ordem lógica:

```bash
# Execução completa de ponta a ponta
python main.py --etapa tudo

# Execução completa com limite por veículo
python main.py --etapa tudo --limite 200
```

---

### 9. Comandos Utilitários de Diagnóstico e Banco de Dados

```bash
# Exibir painel com o status detalhado da base e métricas analíticas de NLP
python main.py --status

# Restaurar eixos temáticos editoriais baseando-se nas URLs originais de coleta
python main.py --restaurar-eixos

# Redefinir todas as URLs com status erro_download de volta para pendente
python main.py --reset-erros

# Redefinir erro_download para pendente apenas de um veículo específico
python main.py --reset-erros --veiculo metropoles

# Limpar arquivos de logs de execuções anteriores
python main.py --limpar-logs

# Resetar completamente o banco de dados (recriar tabelas vazias)
python main.py --reset-db
```

---

## Painel de Diagnóstico (`--status`)

O comando `python main.py --status` fornece um raio-X completo e em tempo real do pipeline:
1. **Resumo Geral da Base:** Contagem de URLs coletadas (CDX vs Sitemaps), status da extração e total de matérias únicas.
2. **Progresso de IA & NLP:** Indicador global de embeddings salvos, matérias com sentimento/clickbait e menções a entidades catalogadas.
3. **Distribuição por Veículo com Cobertura de NLP:** Tabela com total de matérias extraídas, taxa de aproveitamento e porcentagem de conclusão de embeddings por jornal.
4. **Distribuição por Eixo Temático:** Proporção de notícias em Política, Economia, Internacional, Social, Ambiental e Esportivo.
5. **Métricas Consolidadas de NLP:** Índice médio de sensacionalismo (*clickbait*), polaridade média geral de sentimento, distribuição de tom noticioso (*Positivo / Neutro / Negativo*) e dimensões do acervo vetorial.

---

## Conformidade Ética e Anonimização Metodológica

1. **Uso Exclusivamente Acadêmico:** A raspagem e a extração cumprem finalidade estrita de pesquisa científica, fundamentada no art. 46 da Lei de Direitos Autorais (Lei Federal nº 9.610/1998) e no Marco Civil da Internet (Lei Federal nº 12.965/2014).
2. **Boas Práticas de Coleta:** Respeito integral a diretivas `robots.txt`, cabeçalhos informativos identificando o projeto e taxas de requisição rigorosamente espaçadas para evitar sobrecarga em servidores.
3. **Anonimização na Apresentação Científica:** Na consolidação de relatórios e artigos, os veículos são identificados por designadores tipológicos anônimos (*Tradicional 1 a 5* e *Digital Nativo 1 a 5*) para preservar a isenção analítica e o foco na estrutura sistêmica da comunicação.
