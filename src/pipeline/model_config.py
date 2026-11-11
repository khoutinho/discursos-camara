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
    'minilm': EmbeddingSpec(
        label='MiniLM (rápido, funciona sem GPU)',
        descricao='Padrão. Lê o começo de cada discurso (~85 palavras, depois da limpeza). Indicado para '
                  'computador sem GPU e para testes rápidos; para a análise final, prefira o bge-m3.',
        # sem trechos: a média dos trechos deixava os discursos parecidos demais e o HDBSCAN juntava quase
        # tudo em um tópico; 'leaf' evita o mesmo colapso em parte das amostras (testes de nov/2026)
        hf_name='paraphrase-multilingual-MiniLM-L12-v2', max_seq_length=128, chunk_words=None,
        batch_size=64, fp16_on_gpu=False, min_cluster_size=10, cluster_selection_method='leaf',
        ref_sec_per_doc_cpu=0.08, ref_sec_per_doc_gpu=0.005,
    ),
    'bge-m3': EmbeddingSpec(
        label='bge-m3 (mais preciso, requer GPU)',
        descricao='Lê o discurso inteiro (até 8.192 tokens). Gerou os tópicos mais específicos nos testes. '
                  'Indicado para a análise final, com GPU; em CPU é muito lento.',
        hf_name='BAAI/bge-m3', max_seq_length=8192, chunk_words=5000,
        batch_size=2, fp16_on_gpu=True, min_cluster_size=15, cluster_selection_method='eom',
        ref_sec_per_doc_cpu=2.0, ref_sec_per_doc_gpu=0.045,
    ),
}
DEFAULT_EMBEDDING_MODEL = 'minilm'

# min_cluster_size proporcional ao corpus: com valor fixo, o número de tópicos cresce com o corpus
# (o HDBSCAN divide tópicos grandes sempre que cada parte atinge o mínimo)
MIN_CLUSTER_SHARE = 0.01


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

    # Multi-label: threshold de probabilidade mínima pra atribuir um tópico
    multi_label_threshold: float = 0.1

    # Cache (só usado pela linha de comando; o app guarda embeddings no SQLite)
    cache_dir: Path | None = None

    def __post_init__(self):
        if self.embedding_model not in EMBEDDING_MODELS:
            raise ValueError(f"modelo desconhecido: {self.embedding_model}; opções: {', '.join(EMBEDDING_MODELS)}")

    @property
    def spec(self) -> EmbeddingSpec:
        return EMBEDDING_MODELS[self.embedding_model]

    def resolved_min_cluster_size(self, n_docs: int) -> int:
        if self.min_cluster_size is not None:
            return self.min_cluster_size
        return max(self.spec.min_cluster_size, round(n_docs * MIN_CLUSTER_SHARE))


def methodology(cfg: TopicModelConfig, n_docs: int | None = None) -> list[dict]:
    """Parâmetros usados em cada etapa, em formato de tabela (n_docs: discursos analisados, se já se sabe)."""
    reduce = str(cfg.nr_topics) if cfg.nr_topics is not None else 'sem redução'
    spec = cfg.spec
    chunks = (f'{spec.chunk_words} palavras' if spec.chunk_words
              else f'não (lê os primeiros {spec.max_seq_length} tokens)')
    if cfg.min_cluster_size is not None:
        mcs = cfg.min_cluster_size
    elif n_docs is not None:
        mcs = f'{cfg.resolved_min_cluster_size(n_docs)} (máx. entre {spec.min_cluster_size} e 1% de {n_docs} discursos)'
    else:
        mcs = f'máx. entre {spec.min_cluster_size} e 1% dos discursos'
    rows = [
        ('Pré-processamento', 'Limpeza do texto', f'v{CLEANING_VERSION}',
         'Antes do embedding e do vocabulário, remove o cabeçalho do orador, falas da Presidência e apartes, '
         'marcações da sessão (palmas, microfone), vocativos e cumprimentos de abertura, "Era o que tinha a dizer" '
         'e pedidos de divulgação. O texto original fica guardado.'),
        ('Embeddings', 'Modelo', spec.hf_name,
         'Sentence-transformer que converte cada discurso (texto limpo, com acentos) em um vetor numérico.'),
        ('Embeddings', 'max_seq_length', spec.max_seq_length, 'Tokens lidos de cada trecho.'),
        ('Embeddings', 'trechos', chunks,
         'O discurso é dividido em trechos que cabem no modelo; o embedding do discurso é a média dos '
         'embeddings dos trechos (normalizada).'),
        ('UMAP', 'n_neighbors', cfg.umap_n_neighbors,
         'Quantos vizinhos cada discurso considera ao reduzir a dimensão; valores maiores priorizam a estrutura global.'),
        ('UMAP', 'n_components', cfg.umap_n_components, 'Número de dimensões após a redução.'),
        ('UMAP', 'min_dist', cfg.umap_min_dist, 'Distância mínima entre pontos; 0 favorece grupos compactos.'),
        ('UMAP', 'metric', 'cosine', 'Medida de similaridade entre embeddings.'),
        ('UMAP', 'random_state', 42, 'Semente fixa para tornar o resultado reprodutível.'),
        ('HDBSCAN', 'min_cluster_size', mcs,
         'Tamanho mínimo de um tópico; grupos menores viram ruído. Proporcional ao corpus, para o número de '
         'tópicos não crescer só porque a coleta é maior.'),
        ('HDBSCAN', 'cluster_selection_method', spec.cluster_selection_method,
         'Como o HDBSCAN escolhe os tópicos na hierarquia de grupos: "eom" fica com os grupos mais estáveis '
         '(pode juntar quase tudo em um só quando os embeddings são muito parecidos); "leaf" fica com os '
         'grupos mais finos.'),
        ('HDBSCAN', 'min_samples', cfg.min_samples,
         'Quão conservador é o agrupamento; valores baixos classificam menos discursos como ruído.'),
        ('Ruído', 'reduce_outliers', 'embeddings' if cfg.reduce_outliers else 'não',
         'Discursos que o HDBSCAN deixou sem tópico (ruído) são atribuídos ao tópico cujo centróide de embeddings '
         'é mais similar. A coluna Topico_reatribuido do CSV marca esses discursos.'),
        ('Redução de tópicos', 'nr_topics', reduce,
         'Número alvo de tópicos; o BERTopic une os tópicos mais parecidos até chegar nele.'),
        ('Vocabulário (c-TF-IDF)', 'ngram_range', '(1, 2)', 'Considera palavras isoladas e pares de palavras.'),
        ('Vocabulário (c-TF-IDF)', 'min_df', 2, 'Ignora termos que aparecem em menos de 2 tópicos.'),
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
        ('Vocabulário (c-TF-IDF)', 'reduce_frequent_words', True,
         'Diminui o peso de palavras frequentes em todos os tópicos.'),
        ('Rótulos', 'KeyBERTInspired top_n_words', 10,
         'Reordena as palavras de cada tópico pela similaridade com o próprio tópico.'),
        ('Rótulos', 'MaximalMarginalRelevance diversity', 0.3,
         'Evita palavras redundantes no rótulo (ex.: "mulher" e "mulheres").'),
        ('Multi-label', 'threshold', cfg.multi_label_threshold,
         'Probabilidade mínima (pertencimento suave do HDBSCAN a cada tópico) para um discurso receber um '
         'tópico além do principal.'),
    ]
    return [
        {'etapa': e, 'parametro': p, 'valor': str(v), 'descricao': d}
        for e, p, v, d in rows
    ]
