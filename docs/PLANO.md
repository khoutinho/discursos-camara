# Plano de Implementação — Projeto Câmara

## Contexto

Projeto acadêmico de NLP que analisa discursos da Câmara dos Deputados brasileira sobre violência contra a mulher. O projeto atual consiste em:

- **Scraper** (`arquivos/criador_csv.ipynb`): raspa discursos do SitaQWeb (`camara.leg.br/internet/sitaqweb/`) por keywords, salva em CSV com texto normalizado (sem acentos, sem stopwords)
- **Pipeline NLP** (`arquivos/pipeline.py`): BERTopic com k-means + detecção de temas por regex + análises estatísticas
- **Dashboard** (`arquivos/dashboard_v2.html`): visualização Chart.js com 8 abas, carrega `dashboard_data.json`
- **Corpus stats** (`arquivos/noimjes.py`): estatísticas do corpus para o artigo científico

O objetivo é **unificar tudo em uma aplicação local** (FastAPI + HTML/JS) e **melhorar o pipeline NLP** para que o BERTopic produza tópicos interpretáveis, substituindo a detecção de temas por regex.

---

## Fase 1 — Pipeline NLP (prioridade máxima)

### Problemas diagnosticados no pipeline atual

1. **Corpus pré-filtrado homogêneo**: toda frase contém keywords de busca ("violência", "mulher", "contra"), que dominam todos os tópicos
2. **Stopwords de domínio não usadas no BERTopic**: `STOPWORDS_EXTRA` (com "deputado", "presidente", "projeto", "não", "fazer") existe no código mas só é usado na análise de vocabulário, nunca no topic modeling
3. **Nenhum `CountVectorizer` customizado** passado ao BERTopic — c-TF-IDF não filtra palavras genéricas
4. **Embeddings gerados sobre texto lematizado/normalizado** — modelo foi treinado em texto natural, recebe tokens processados
5. **K-Means força toda frase num cluster** — frases genéricas/boilerplate não podem ir pra ruído
6. **Topic modeling por frase** — perde contexto do discurso inteiro
7. **Labels são top-5 palavras cruas** do c-TF-IDF sem pós-processamento
8. **spaCy `pt_core_news_sm`** produz erros de lematização (ex: "violencer" em vez de "violência")

### Decisões de design do novo pipeline

| Decisão | Escolha |
|---------|---------|
| Classificação temática | BERTopic como único sistema (substitui regex) |
| Granularidade | Discurso inteiro (não frase) |
| Multi-label | Sim — cada discurso pode pertencer a múltiplos tópicos via probabilidades |
| Clustering | HDBSCAN (descobre nº de tópicos automaticamente) + `reduce_topics()` pra simplificar |
| Vocabulário do c-TF-IDF | `CountVectorizer(stop_words=keywords_de_busca + stopwords_parlamentares)` — remove termos ubíquos |
| Embedding model | Manter `paraphrase-multilingual-MiniLM-L12-v2` por agora; testar outros depois se necessário |
| Input do embedding | Texto original (com acentos, natural) — normalização só pro CountVectorizer |
| Representação dos tópicos | KeyBERTInspired + MaximalMarginalRelevance (+ LLM opcional no futuro) |
| Cache de embeddings | Sim — só recalcula se modelo de embedding mudar |
| Classificação ideológica | Manter Bolognesi, Ribeiro & Codato (2023), 7 classes + 3 classes; verificar partidos pós-2023 |

### Fluxo do novo pipeline

```
Texto original (com acentos)
    │
    ├──► Embedding model ──► vetores 384-dim (um por discurso)
    │                              │
    │                              ├──► UMAP (redução dimensional)
    │                              │
    │                              ├──► HDBSCAN (clustering)
    │                              │
    │                              └──► reduce_topics() (simplificação)
    │
    └──► Normalização (lowercase, remove pontuação, lematização)
              │
              └──► CountVectorizer (com stopwords customizadas)
                        │
                        └──► c-TF-IDF ──► KeyBERT + MMR ──► labels dos tópicos
```

Duas entradas separadas pro BERTopic:
1. **Embeddings**: gerados sobre texto original/natural
2. **Representação**: gerada sobre texto normalizado via CountVectorizer customizado

### Análises estatísticas

Manter existentes:
- Mann-Kendall (tendência temporal)
- Correlação de Pearson
- Similaridade de cosseno entre espectros
- Coeficiente de variação por partido

Adicionar (sem redundância):
- Evolução temporal por tópico BERTopic
- Correlação tópico ↔ espectro ideológico

### Tarefas da Fase 1

1. Criar novo `pipeline.py` com as decisões acima
2. Testar com os CSVs existentes (texto normalizado — resultados subótimos mas válidos pra validação)
3. Avaliar qualidade dos tópicos — se genéricos, testar:
   - Outros modelos de embedding (ex: `paraphrase-multilingual-mpnet-base-v2`, modelos portugueses)
   - Ajuste de `min_cluster_size` do HDBSCAN
   - Re-scraping com texto original
4. Iterar até tópicos interpretáveis

### Por que embeddings sobre texto original importam

Modelos de embedding (sentence-transformers) foram treinados em bilhões de frases naturais com acentos, pontuação, artigos, preposições. Quando recebem texto normalizado ("violencia domestica problema grave afetar mulher"), não conseguem usar contexto gramatical pra desambiguar significados. O texto original preserva a qualidade dos embeddings.

A normalização (stopwords, lematização) serve apenas pra etapas de contagem de palavras (CountVectorizer → c-TF-IDF), onde queremos que "mulheres" e "mulher" contem como o mesmo token.

---

## Fase 2 — Benchmark de Scraping

### Contexto

O scraper atual usa screen-scraping do SitaQWeb (sistema legado da Câmara). A API oficial "Dados Abertos" (`dadosabertos.camara.leg.br/api/v2`) retorna texto original no campo `transcricao`, mas não tem busca global por keyword — precisa iterar por deputado.

### Os 3 métodos a testar

| Método | Como funciona | Prós | Contras |
|--------|--------------|------|---------|
| **SitaQWeb puro** | Busca por keyword direto no HTML | Rápido, filtra na fonte | HTML scraping frágil, texto já processado |
| **API pura** | Lista ~515 deputados → puxa discursos de cada um → filtra por keyword em Python | Dados ricos, texto original, estável | Lento (iterar por deputado), sem busca global |
| **Híbrido** | SitaQWeb descobre quais deputados discursaram → API puxa textos desses deputados → filtra por keyword | Combina velocidade com qualidade | Mais complexo, depende dos dois sistemas |

### Teste

- Usar um tema com poucos discursos (ex: um subtema específico ou período curto)
- Medir: tempo total, nº de discursos encontrados, qualidade do texto
- Decidir abordagem definitiva com base nos resultados

### API Dados Abertos — referência técnica

- **Discursos**: `GET /deputados/{id}/discursos?dataInicio=YYYY-MM-DD&dataFim=YYYY-MM-DD&itens=100&pagina=N`
- **Campos relevantes**: `transcricao` (texto completo), `keywords` (temas oficiais), `sumario`, `tipoDiscurso`, `faseEvento`, `dataHoraInicio`
- **Lista de deputados**: `GET /deputados?itens=100&pagina=N` (~515 deputados, ~6 páginas)
- **Paginação**: max 100 itens/página, header `X-Total-Count` com total
- **Rate limit**: header `Retry-After: 30` presente em todas as respostas

### SitaQWeb — referência técnica

- **URL**: `https://www.camara.leg.br/internet/sitaqweb/resultadoPesquisaDiscursos.asp`
- **Busca**: por keyword, range de data, tipo de sessão (`basePesq=plenario`)
- **Parsing**: BeautifulSoup, tabela de resultados → links individuais → `<p>` após `<hr>`
- **Paginação**: link "Proxima" na página de resultados

---

## Fase 3 — Aplicação Unificada

### Tech stack

- **Backend**: FastAPI (Python) — WebSocket pra progress bars, endpoints REST pro scraping e pipeline
- **Frontend**: HTML/JS com Chart.js — evolução do `dashboard_v2.html` existente
- **Banco**: SQLite (interno) — scraping incremental, cache de embeddings, resultados
- **Exportação**: CSV/JSON dos dados, PNG/SVG dos gráficos

### UX — Duas telas

**Tela 1: Configuração & Processamento**

Home com 3 entry points:
1. **"Criar do zero"** → configura keywords + range de data → scraping com barra de progresso → pipeline com barra de progresso → vai pro dashboard
2. **"Já possuo CSV"** → importa CSV → pipeline com barra de progresso → vai pro dashboard
3. **"Já possuo dashboard_data"** → importa JSON → vai direto pro dashboard

Controles do pipeline (com tooltips explicativos):
- Número alvo de tópicos (`reduce_topics`) — default automático
- Modelo de embedding — dropdown com opções disponíveis
- Threshold de probabilidade pra multi-label — slider (default: 0.3)

Painel de documentação acessível: explicação técnica de cada parâmetro, exportável como tabela pro paper.

**Tela 2: Dashboard**

8 abas (adaptadas do existente):
1. **Visão Geral** — evolução temporal, distribuição por espectro, top partidos, fases
2. **Esquerda x Direita** — proporção de espectros ao longo do tempo
3. **Tópicos** — resultados do BERTopic (substitui "Temas"), tópicos por espectro/tempo
4. **Temas** → renomeado pra **Tópicos** ou removido se redundante
5. **Regiões** — distribuição geográfica
6. **Partidos** — análise por partido
7. **Vocabulário** — top words por espectro
8. **Estatísticas** — Mann-Kendall, correlação, coeficiente de variação + novas análises por tópico

Cada gráfico com botão de exportação PNG/SVG.

### Dados e persistência

- **SQLite**: armazena discursos raspados, embeddings cacheados, resultados do pipeline
- **Etapas independentes**: scraping → embeddings → clustering → visualização — cada uma cacheada
- **Re-processamento**: mudar parâmetros do clustering não recalcula embeddings
- **Scraping**: paralelo (ex: 5 requisições simultâneas) com rate limiting, retomável se interrompido (SQLite registra progresso)

### Formato do CSV aceito (path "Já possuo CSV")

Apenas novo formato: texto original com acentos. Colunas esperadas:
`Data, Sessao, Orador, Partido, Estado, Fase, Hora, Discurso`

### Encoding

Sempre usar `utf-8-sig` na leitura e escrita de CSVs. O BOM (byte order mark) garante que Excel e outras ferramentas reconheçam acentos corretamente. O pipeline anterior removia acentos (NFKD → ASCII) pra contornar problemas de encoding — a solução correta é acertar o encoding, não destruir os dados.

---

## Decisões pendentes

- Resultado do benchmark de scraping (Fase 2) pode mudar a abordagem de scraping definitiva
- Se tópicos do BERTopic não ficarem bons mesmo após melhorias, considerar:
  - Trocar modelo de embedding
  - Reintroduzir keywords no vocabulário (fallback)
  - Abordagem híbrida BERTopic + regex
- Modelo de LLM pra labels (se adicionado no futuro): qual usar, custo, dependência de API

---

## Referências

- Classificação ideológica: Bolognesi, Ribeiro & Codato (2023)
- Metodologia base: "Mudanças Climáticas no Discurso Parlamentar Brasileiro" (2025)
- BERTopic: Grootendorst (2022), https://maartengr.github.io/BERTopic/
- Embedding model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
