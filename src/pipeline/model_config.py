"""
Parâmetros do topic modeling e a tabela de metodologia (para o paper).

Fica separado de topic_modeling.py porque não depende de PyTorch/BERTopic:
o app consegue mostrar os parâmetros sem carregar os modelos.
"""

from dataclasses import dataclass
from pathlib import Path

from .config import STOPWORDS_DOMAIN
from .text_cleaning import CLEANING_VERSION
from .vocabulary import SPACY_MODEL, UBIQUITOUS_SHARE

@dataclass(frozen=True)
class EmbeddingSpec:
    """Como gerar os embeddings com um modelo, e os parâmetros que funcionaram com ele."""

    label: str
    descricao: str
    hf_name: str               # nome no Hugging Face
    max_seq_length: int        # tokens lidos por trecho
    chunk_words: int | None    # discurso dividido em trechos desse tamanho (média dos embeddings); None = inteiro
    batch_size: int
    fp16_on_gpu: bool
    min_cluster_size: int      # piso do min_cluster_size (o valor usado é max(piso, 1% dos discursos))
    cluster_selection_method: str  # 'eom' ou 'leaf' (HDBSCAN); ver testes de nov/2026 no PENDENCIAS
    # segundos por discurso medidos na máquina de desenvolvimento (RTX 4070 Ti / CPU de 20 threads);
    # usados na estimativa de tempo até o app medir a máquina de quem está usando
    ref_sec_per_doc_cpu: float
    ref_sec_per_doc_gpu: float


EMBEDDING_MODELS = {
    'bge-m3': EmbeddingSpec(
        label='bge-m3 (mais preciso, requer GPU)',
        descricao='Padrão. Lê o discurso inteiro (até 8.192 tokens). Gerou os tópicos mais específicos e '
                  'estáveis nos testes. Indicado para a análise final, com GPU; em CPU é muito lento (use o MiniLM).',
        hf_name='BAAI/bge-m3', max_seq_length=8192, chunk_words=5000,
        batch_size=2, fp16_on_gpu=True, min_cluster_size=15, cluster_selection_method='eom',
        ref_sec_per_doc_cpu=2.0, ref_sec_per_doc_gpu=0.045,
    ),
    'minilm': EmbeddingSpec(
        label='MiniLM (rápido, funciona sem GPU)',
        descricao='Lê o começo de cada discurso (~85 palavras, depois da limpeza). Indicado para computador '
                  'sem GPU e para testes rápidos; para a análise final, prefira o bge-m3.',
        # sem trechos: a média dos trechos deixava os discursos parecidos demais e o HDBSCAN juntava quase
        # tudo em um tópico; 'leaf' evita o mesmo colapso em parte das amostras (testes de nov/2026)
        hf_name='paraphrase-multilingual-MiniLM-L12-v2', max_seq_length=128, chunk_words=None,
        batch_size=64, fp16_on_gpu=False, min_cluster_size=10, cluster_selection_method='leaf',
        ref_sec_per_doc_cpu=0.08, ref_sec_per_doc_gpu=0.005,
    ),
}
DEFAULT_EMBEDDING_MODEL = 'bge-m3'

# min_cluster_size proporcional ao corpus: com valor fixo, o número de tópicos cresce com o corpus
# (o HDBSCAN divide tópicos grandes sempre que cada parte atinge o mínimo)
MIN_CLUSTER_SHARE = 0.01

# rodadas do consenso: 15 dão o mesmo resultado que 30 (testes de nov/2026, ver consensus.py)
CONSENSUS_RUNS = 15
CONSENSUS_SEED = 42      # sementes 42, 43, …; uma rodada só usa a 42
CONSENSUS_CUT = 0.5      # tópico = discursos juntos em mais desta fração das rodadas

# vocabulário e rótulos (usados em topic_modeling.py e documentados em methodology())
NGRAM_RANGE = (1, 2)
MIN_DF = 2                 # termo precisa aparecer em ≥2 tópicos
KEYBERT_TOP_N = 10
MMR_DIVERSITY = 0.3
LABEL_WORDS = 5            # palavras no rótulo de cada tópico
LABEL_MAX_SEQ_LENGTH = 512  # contexto do modelo usado só para escolher os rótulos (KeyBERTInspired)


def embedding_cache_key(model_key: str) -> str:
    """Chave do cache de embeddings: muda quando o modelo, os trechos ou a limpeza do texto mudam."""
    spec = EMBEDDING_MODELS[model_key]
    return f"{spec.hf_name}|seq{spec.max_seq_length}|chunks{spec.chunk_words}|clean-v{CLEANING_VERSION}"


@dataclass
class TopicModelConfig:
    """Parâmetros configuráveis do topic modeling."""

    embedding_model: str = DEFAULT_EMBEDDING_MODEL  # chave de EMBEDDING_MODELS

    # HDBSCAN (min_cluster_size None = max(piso do modelo, 1% dos discursos), calculado no fit)
    min_cluster_size: int | None = None
    min_samples: int = 1

    # Discursos que o HDBSCAN deixa como ruído vão para o tópico de centróide mais próximo
    reduce_outliers: bool = True

    # UMAP
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.0

    # reduce_topics: None = sem redução (todos os tópicos do HDBSCAN), int = número alvo
    nr_topics: int | None = None

    # Tópicos de consenso: rodadas de UMAP → HDBSCAN com sementes diferentes (consensus.py).
    # 1 = uma rodada só (semente 42), como antes de nov/2026.
    consensus_runs: int = CONSENSUS_RUNS

    # Multi-label: com consenso, fração mínima das rodadas em que o discurso fica junto de outro tópico;
    # sem consenso, probabilidade mínima do HDBSCAN (pertencimento suave)
    multi_label_threshold: float = 0.2

    # Cache (só usado pela linha de comando; o app guarda embeddings no SQLite)
    cache_dir: Path | None = None

    def __post_init__(self):
        if self.embedding_model not in EMBEDDING_MODELS:
            raise ValueError(f"modelo desconhecido: {self.embedding_model}; opções: {', '.join(EMBEDDING_MODELS)}")
        if self.consensus_runs < 1:
            raise ValueError("consensus_runs precisa ser ≥ 1 (1 = uma rodada só, sem consenso)")

    @property
    def consensus(self) -> bool:
        return self.consensus_runs > 1

    @property
    def spec(self) -> EmbeddingSpec:
        return EMBEDDING_MODELS[self.embedding_model]

    def resolved_min_cluster_size(self, n_docs: int) -> int:
        if self.min_cluster_size is not None:
            return self.min_cluster_size
        return max(self.spec.min_cluster_size, round(n_docs * MIN_CLUSTER_SHARE))


def _num(v):
    """Número no formato do português (0,2), para a tabela ir direto para o paper."""
    return str(v).replace('.', ',') if isinstance(v, float) else v


def methodology(cfg: TopicModelConfig, n_docs: int | None = None) -> list[dict]:
    """
    Parâmetros de cada etapa, da coleta às estatísticas, em formato de tabela (para o paper).

    n_docs: discursos analisados, se já se sabe (mostra o min_cluster_size calculado).
    """
    # leve: só constantes; trends/preprocessing não carregam PyTorch
    from .config import ESPECTRO
    from .trends import MIN_MONTHS_SEASONAL, MIN_YEARS_ANNUAL

    spec = cfg.spec
    reduce = str(cfg.nr_topics) if cfg.nr_topics is not None else 'sem redução'
    chunks = (f'{spec.chunk_words} palavras' if spec.chunk_words
              else f'não (lê os primeiros {spec.max_seq_length} tokens)')
    if cfg.min_cluster_size is not None:
        mcs = cfg.min_cluster_size
    elif n_docs is not None:
        mcs = (f'{cfg.resolved_min_cluster_size(n_docs)} (máx. entre {spec.min_cluster_size} e 1% de '
               f'{n_docs:,} discursos)'.replace(',', '.'))
    else:
        mcs = f'máx. entre {spec.min_cluster_size} e 1% dos discursos'
    per_round = ' Roda em cada rodada do consenso.' if cfg.consensus else ''
    if cfg.consensus:
        seeds = f'{CONSENSUS_SEED}–{CONSENSUS_SEED + cfg.consensus_runs - 1}'
        seeds_desc = 'Uma semente fixa por rodada do consenso, para o resultado ser reprodutível.'
    else:
        seeds, seeds_desc = CONSENSUS_SEED, 'Semente fixa para tornar o resultado reprodutível.'

    rows = [
        ('Coleta', 'fonte', 'SitaQWeb (Câmara dos Deputados)',
         'Busca no texto das notas taquigráficas do Plenário (inclui as sessões de Comissão Geral). Keywords com '
         'acento são buscadas também sem acento, porque o índice do SitaQWeb mudou no fim de 2022 (sem acento, a '
         'busca só acha discursos de 2023 em diante). Antes de 2000, só parte dos discursos aparece na busca.'),
        ('Pré-processamento', 'limpeza do texto', f'v{CLEANING_VERSION}',
         'Antes do embedding e do vocabulário, fica só a fala do orador: saem o cabeçalho, as marcações de '
         'discurso entregue por escrito, as falas da Presidência e os apartes, as marcações da sessão (palmas, '
         'microfone), os vocativos e cumprimentos de abertura, "Era o que tinha a dizer" e os pedidos de '
         'divulgação. O texto original fica guardado.'),
        ('Embeddings', 'modelo', spec.hf_name,
         'Sentence-transformer que converte cada discurso (texto limpo, com acentos) em um vetor numérico.'),
        ('Embeddings', 'max_seq_length', spec.max_seq_length, 'Tokens lidos de cada trecho.'),
        ('Embeddings', 'trechos', chunks,
         'O discurso é dividido em trechos que cabem no modelo; o embedding do discurso é a média dos '
         'embeddings dos trechos (normalizada).'),
        ('UMAP', 'n_neighbors', cfg.umap_n_neighbors,
         'Quantos vizinhos cada discurso considera ao reduzir a dimensão; valores maiores priorizam a estrutura '
         'global.' + per_round),
        ('UMAP', 'n_components', cfg.umap_n_components, 'Número de dimensões após a redução.'),
        ('UMAP', 'min_dist', cfg.umap_min_dist, 'Distância mínima entre pontos; 0 favorece grupos compactos.'),
        ('UMAP', 'metric', 'cosine', 'Medida de similaridade entre embeddings.'),
        ('UMAP', 'random_state', seeds, seeds_desc),
        ('HDBSCAN', 'min_cluster_size', mcs,
         'Tamanho mínimo de um tópico; grupos menores viram ruído. Proporcional ao corpus, para o número de '
         'tópicos não crescer só porque a coleta é maior.' + per_round),
        ('HDBSCAN', 'cluster_selection_method', spec.cluster_selection_method,
         'Como o HDBSCAN escolhe os tópicos na hierarquia de grupos: "eom" fica com os grupos mais estáveis '
         '(pode juntar quase tudo em um só quando os embeddings são muito parecidos); "leaf" fica com os '
         'grupos mais finos.'),
        ('HDBSCAN', 'min_samples', cfg.min_samples,
         'Quão conservador é o agrupamento; valores baixos classificam menos discursos como ruído.'),
    ]
    if cfg.consensus:
        rows += [
            ('Consenso', 'rodadas', cfg.consensus_runs,
             'UMAP → HDBSCAN rodam com uma semente diferente em cada rodada. Para cada par de discursos, conta-se '
             'a fração das rodadas em que os dois ficaram no mesmo grupo (matriz de coassociação; ruído não conta '
             'como junto). Com uma rodada só, parte dos discursos muda de tópico quando a semente muda.'),
            ('Consenso', 'agrupamento', f'ligação média, corte em {_num(CONSENSUS_CUT)}',
             f'Os tópicos são os grupos de discursos que ficam juntos em mais de {CONSENSUS_CUT:.0%} das rodadas '
             '(agrupamento hierárquico sobre 1 − coassociação). Grupos menores que min_cluster_size ficam sem '
             'tópico. O BERTopic recebe esses tópicos prontos e só calcula os rótulos.'),
            ('Consenso', 'certeza', 'coluna Topico_certeza',
             'Fração das rodadas em que o discurso ficou junto dos discursos do próprio tópico. Baixa nos '
             'discursos sem consenso levados ao tópico mais próximo.'),
        ]
    rows += [
        ('Discursos sem tópico', 'reduce_outliers', 'embeddings' if cfg.reduce_outliers else 'não',
         f'Discursos sem tópico ({"sem consenso" if cfg.consensus else "ruído do HDBSCAN"}) vão para o tópico cujo '
         'centróide de embeddings é mais similar. A coluna Topico_reatribuido do CSV marca esses discursos.'),
        ('Redução de tópicos', 'nr_topics', reduce,
         'Número alvo de tópicos; o BERTopic une os tópicos mais parecidos até chegar nele.'),
        ('Vocabulário (c-TF-IDF)', 'ngram_range', str(NGRAM_RANGE), 'Considera palavras isoladas e pares de palavras.'),
        ('Vocabulário (c-TF-IDF)', 'tokens', 'palavras com 2+ letras',
         'Números ficam de fora. Verbos com pronome oblíquo ("criticá-la") e pares da mesma palavra repetida '
         '("democracia democracia") também.'),
        ('Vocabulário (c-TF-IDF)', 'min_df', MIN_DF, f'Ignora termos que aparecem em menos de {MIN_DF} tópicos.'),
        ('Vocabulário (c-TF-IDF)', 'stopwords', f'{len(STOPWORDS_DOMAIN)} termos',
         'Stopwords do português (NLTK) e jargão parlamentar.'),
        ('Vocabulário (c-TF-IDF)', 'palavras muito frequentes', f'em ≥{UBIQUITOUS_SHARE:.0%} dos discursos',
         'Saem do vocabulário, com as variações de acento e plural: aparecem em quase todos os tópicos e não '
         'diferenciam nenhum. Em geral são a keyword de uma busca com um termo só e o vocabulário que a '
         'acompanha (ex.: "mulher" e "violência" na busca por "feminicídio"); numa busca com várias '
         'keywords, as que aparecem em menos da metade dos discursos podem aparecer nos rótulos.'),
        ('Vocabulário (c-TF-IDF)', 'classes gramaticais', 'substantivo, nome próprio, adjetivo, particípio',
         f'Só palavras de conteúdo entram nos rótulos: a classe mais comum de cada palavra no corpus é '
         f'identificada com o spaCy ({SPACY_MODEL}); verbos, advérbios, pronomes e números ficam de fora.'),
        ('Vocabulário (c-TF-IDF)', 'reduce_frequent_words', 'sim',
         'Diminui o peso de palavras frequentes em todos os tópicos.'),
        ('Rótulos', 'KeyBERTInspired top_n_words', KEYBERT_TOP_N,
         'Reordena as palavras de cada tópico pela similaridade com o próprio tópico (no mesmo modelo de '
         f'embedding, lendo até {LABEL_MAX_SEQ_LENGTH} tokens).'),
        ('Rótulos', 'MaximalMarginalRelevance diversity', MMR_DIVERSITY,
         'Reduz palavras redundantes no rótulo; variações da mesma palavra ainda podem aparecer '
         '("assassinada | assassinadas").'),
        ('Rótulos', 'palavras no rótulo', LABEL_WORDS, f'O rótulo de cada tópico são as {LABEL_WORDS} primeiras palavras.'),
        ('Multi-label', 'threshold', _num(cfg.multi_label_threshold),
         'Fração mínima das rodadas do consenso em que o discurso ficou junto dos discursos de outro tópico para '
         'receber esse tópico além do principal (coluna Topicos_multi). Abaixo de 0,2 o 2º tópico quase nunca se '
         'repetia com outras sementes (testes de nov/2026).' if cfg.consensus else
         'Probabilidade mínima (pertencimento suave do HDBSCAN a cada tópico) para um discurso receber um '
         'tópico além do principal (coluna Topicos_multi).'),
        ('Menções', 'contagem por tópico', 'texto limpo, sem acento, plural opcional',
         'Quantas vezes cada discurso cita as keywords da busca (palavra inteira); o dashboard mostra a mediana e '
         'o % com uma menção só por tópico. Nenhum discurso é excluído por isso.'),
        ('Classificação', 'espectro político', 'Bolognesi, Ribeiro & Codato (2023)',
         f'{len(ESPECTRO)} siglas em 7 classes, agrupadas em Esquerda, Centro e Direita. Grafias da mesma sigla são '
         'unidas; partidos fora da classificação recebem a classe do partido em que se transformaram, e PSDC/DC '
         '(Centro) e Missão (Direita) uma classificação dos autores. Convidados (quem fala sem ser deputado) ficam '
         'no corpus, mas fora dos espectros.'),
        ('Tendências', 'teste', 'Mann-Kendall anual e sazonal',
         f'Anual: sobre os anos completos (precisa de {MIN_YEARS_ANNUAL}). Sazonal (Hirsch & Slack, 1984): compara '
         f'cada mês só com o mesmo mês de outros anos (precisa de {MIN_MONTHS_SEASONAL} meses). A série começa no '
         'mês do primeiro discurso da coleta (janeiro, se ele for de fevereiro: recesso).'),
        ('Tendências', 'séries', 'contagem, proporção, proporção ÷ bancada',
         'Proporção: discursos ÷ total de discursos do Plenário no mês (SitaQWeb). Proporção ÷ bancada (só '
         'espectros): a proporção dividida pela fatia do espectro nas cadeiras na posse de cada legislatura '
         '(API Dados Abertos).'),
        ('Estatísticas', 'correlação entre espectros', 'Pearson e cosseno',
         f'Sobre a contagem anual de cada espectro nos anos completos (precisa de {MIN_YEARS_ANNUAL}).'),
        ('Estatísticas', 'coeficiente de variação', 'desvio padrão ÷ média',
         'Da contagem anual de discursos dos 15 partidos com mais discursos: baixo = participação estável.'),
    ]
    return [
        {'etapa': e, 'parametro': p, 'valor': str(_num(v)), 'descricao': d}
        for e, p, v, d in rows
    ]
