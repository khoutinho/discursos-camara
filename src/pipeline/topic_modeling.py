"""
Topic modeling com BERTopic: embeddings, HDBSCAN, representação KeyBERT+MMR.

Fluxo:
1. Texto original (natural, com acentos) → sentence-transformer → embeddings
2. UMAP reduz dimensionalidade dos embeddings
3. HDBSCAN descobre clusters (tópicos) automaticamente
4. reduce_topics() simplifica para um número gerenciável
5. CountVectorizer com stopwords de domínio → c-TF-IDF → labels
6. KeyBERT + MMR refinam as labels pra diversidade e relevância
7. Probabilidades por tópico permitem multi-label por discurso
"""

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from .config import STOPWORDS_DOMAIN

warnings.filterwarnings('ignore')


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

    # reduce_topics: None = automático (sem redução), int = número alvo
    nr_topics: int | None = None

    # Multi-label: threshold de probabilidade mínima pra atribuir um tópico
    multi_label_threshold: float = 0.1

    # Cache
    cache_dir: Path | None = None


class TopicModeler:
    """Encapsula o pipeline de topic modeling."""

    def __init__(self, config: TopicModelConfig | None = None):
        self.config = config or TopicModelConfig()
        self.model: BERTopic | None = None
        self.embeddings: np.ndarray | None = None

    def fit(self, documents: pd.Series, progress_callback=None) -> dict:
        """
        Roda o pipeline completo de topic modeling.

        Args:
            documents: Series com o texto original (natural, com acentos) de cada discurso.
            progress_callback: Callable(stage: str, pct: float) opcional pra barra de progresso.

        Returns:
            dict com:
                - topics: array de tópico dominante por documento
                - probabilities: matrix de probabilidades por tópico
                - topic_labels: dict {topic_id: "label descritiva"}
                - multi_labels: list de listas — tópicos atribuídos a cada doc (multi-label)
                - model: o objeto BERTopic treinado
        """
        cfg = self.config
        _cb = progress_callback or (lambda *a: None)

        # ── 1. Embeddings sobre texto original ──
        _cb('embeddings', 0.0)
        embedder = SentenceTransformer(cfg.embedding_model)
        embeddings = self._get_or_compute_embeddings(documents, embedder)
        _cb('embeddings', 1.0)

        # ── 2. Componentes do BERTopic ──
        _cb('model', 0.0)

        umap_model = UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            n_components=cfg.umap_n_components,
            min_dist=cfg.umap_min_dist,
            metric='cosine',
            random_state=42,
        )

        hdbscan_model = HDBSCAN(
            min_cluster_size=cfg.min_cluster_size,
            min_samples=cfg.min_samples,
            metric='euclidean',
            prediction_data=True,
        )

        vectorizer = CountVectorizer(
            stop_words=list(STOPWORDS_DOMAIN),
            min_df=2,
            ngram_range=(1, 2),
        )

        ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

        representation = [
            KeyBERTInspired(top_n_words=10),
            MaximalMarginalRelevance(diversity=0.3),
        ]

        self.model = BERTopic(
            embedding_model=embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            ctfidf_model=ctfidf,
            representation_model=representation,
            language='portuguese',
            verbose=True,
            min_topic_size=cfg.min_cluster_size,
        )

        # ── 3. Fit ──
        docs_list = documents.tolist()
        topics, probs = self.model.fit_transform(docs_list, embeddings)

        _cb('model', 0.5)

        # ── 4. Reduzir tópicos (se configurado) ──
        if cfg.nr_topics is not None:
            self.model.reduce_topics(docs_list, nr_topics=cfg.nr_topics)
            topics = self.model.topics_
            probs = self.model.probabilities_

        _cb('model', 0.8)

        # ── 5. Labels ──
        topic_labels = self._extract_labels()

        # ── 6. Multi-label ──
        multi_labels = self._assign_multi_labels(probs, topics, topic_labels)

        _cb('model', 1.0)

        self.embeddings = embeddings

        return {
            'topics': np.array(topics),
            'probabilities': probs,
            'topic_labels': topic_labels,
            'multi_labels': multi_labels,
            'model': self.model,
        }

    def _get_or_compute_embeddings(
        self, documents: pd.Series, embedder: SentenceTransformer,
    ) -> np.ndarray:
        """Gera embeddings ou carrega do cache."""
        cfg = self.config
        cache_path = None

        if cfg.cache_dir:
            cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            model_slug = cfg.embedding_model.replace('/', '_')
            cache_path = cfg.cache_dir / f'embeddings_{model_slug}_{len(documents)}.npy'
            if cache_path.exists():
                print(f"Carregando embeddings do cache: {cache_path}")
                return np.load(cache_path)

        print(f"Gerando embeddings com {cfg.embedding_model}...")
        embeddings = embedder.encode(
            documents.tolist(),
            show_progress_bar=True,
            batch_size=32,
        )

        if cache_path:
            np.save(cache_path, embeddings)
            print(f"Embeddings salvos em cache: {cache_path}")

        return embeddings

    def _extract_labels(self) -> dict[int, str]:
        """Extrai labels legíveis dos tópicos."""
        labels = {}
        for topic_id in self.model.get_topics():
            if topic_id == -1:
                labels[-1] = 'Ruído / Genérico'
                continue
            words = self.model.get_topic(topic_id)
            if words:
                top_words = ' | '.join(w for w, _ in words[:5])
                labels[topic_id] = f"T{topic_id}: {top_words}"
        return labels

    def _assign_multi_labels(
        self,
        probs: np.ndarray,
        topics: list[int],
        labels: dict[int, str],
    ) -> list[list[str]]:
        """Atribui múltiplos tópicos por documento baseado no threshold de probabilidade."""
        threshold = self.config.multi_label_threshold
        multi = []

        if probs is None or probs.ndim != 2:
            for t in topics:
                multi.append([labels.get(t, f'T{t}')] if t != -1 else [])
            return multi

        topic_ids = sorted(t for t in labels if t != -1)

        for i in range(len(probs)):
            doc_labels = []
            for j, tid in enumerate(topic_ids):
                if j < probs.shape[1] and probs[i, j] >= threshold:
                    doc_labels.append(labels[tid])
            if not doc_labels and topics[i] != -1:
                doc_labels = [labels.get(topics[i], f'T{topics[i]}')]
            multi.append(doc_labels)

        return multi

    def get_topic_summary(self) -> pd.DataFrame:
        """Retorna resumo dos tópicos pra visualização."""
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        return self.model.get_topic_info()
