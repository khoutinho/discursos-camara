"""
Parâmetros do topic modeling e a tabela de metodologia (para o paper).

Fica separado de topic_modeling.py porque não depende de PyTorch/BERTopic:
o app consegue mostrar os parâmetros sem carregar os modelos.
"""

from dataclasses import dataclass
from pathlib import Path

from .config import STOPWORDS_DOMAIN

EMBEDDING_MODELS = {
    'paraphrase-multilingual-MiniLM-L12-v2': 'Padrão. Rápido, bom para português.',
    'paraphrase-multilingual-mpnet-base-v2': 'Mais preciso, cerca de 3x mais lento.',
}


@dataclass
class TopicModelConfig:
    """Parâmetros configuráveis do topic modeling."""

    embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2'

    # HDBSCAN
    min_cluster_size: int = 10
    min_samples: int = 1

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


def methodology(cfg: TopicModelConfig) -> list[dict]:
    """Parâmetros usados em cada etapa, em formato de tabela."""
    reduce = str(cfg.nr_topics) if cfg.nr_topics is not None else 'sem redução'
    rows = [
        ('Embeddings', 'Modelo', cfg.embedding_model,
         'Sentence-transformer que converte cada discurso (texto original, com acentos) em um vetor numérico. '
         'Lê os primeiros 128 tokens de cada texto.'),
        ('UMAP', 'n_neighbors', cfg.umap_n_neighbors,
         'Quantos vizinhos cada discurso considera ao reduzir a dimensão; valores maiores priorizam a estrutura global.'),
        ('UMAP', 'n_components', cfg.umap_n_components, 'Número de dimensões após a redução.'),
        ('UMAP', 'min_dist', cfg.umap_min_dist, 'Distância mínima entre pontos; 0 favorece grupos compactos.'),
        ('UMAP', 'metric', 'cosine', 'Medida de similaridade entre embeddings.'),
        ('UMAP', 'random_state', 42, 'Semente fixa para tornar o resultado reprodutível.'),
        ('HDBSCAN', 'min_cluster_size', cfg.min_cluster_size,
         'Tamanho mínimo de um tópico; grupos menores viram ruído.'),
        ('HDBSCAN', 'min_samples', cfg.min_samples,
         'Quão conservador é o agrupamento; valores baixos classificam menos discursos como ruído.'),
        ('Redução de tópicos', 'nr_topics', reduce,
         'Número alvo de tópicos; o BERTopic une os tópicos mais parecidos até chegar nele.'),
        ('Vocabulário (c-TF-IDF)', 'ngram_range', '(1, 2)', 'Considera palavras isoladas e pares de palavras.'),
        ('Vocabulário (c-TF-IDF)', 'min_df', 2, 'Ignora termos que aparecem em menos de 2 tópicos.'),
        ('Vocabulário (c-TF-IDF)', 'stopwords', f'{len(STOPWORDS_DOMAIN)} termos',
         'Stopwords do português (NLTK), jargão parlamentar e as keywords da busca.'),
        ('Vocabulário (c-TF-IDF)', 'reduce_frequent_words', True,
         'Diminui o peso de palavras frequentes em todos os tópicos.'),
        ('Rótulos', 'KeyBERTInspired top_n_words', 10,
         'Reordena as palavras de cada tópico pela similaridade com o próprio tópico.'),
        ('Rótulos', 'MaximalMarginalRelevance diversity', 0.3,
         'Evita palavras redundantes no rótulo (ex.: "mulher" e "mulheres").'),
        ('Multi-label', 'threshold', cfg.multi_label_threshold,
         'Probabilidade mínima para um discurso receber um tópico além do principal.'),
    ]
    return [
        {'etapa': e, 'parametro': p, 'valor': str(v), 'descricao': d}
        for e, p, v, d in rows
    ]
