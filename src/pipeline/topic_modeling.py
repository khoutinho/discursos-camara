"""
Topic modeling com BERTopic: embeddings, HDBSCAN, representação KeyBERT+MMR.

Fluxo:
1. Texto limpo (text_cleaning: sem cabeçalho, vocativos, falas de outros) → sentence-transformer → embeddings
   (discurso dividido em trechos quando o modelo lê pouco; embedding = média dos trechos)
2. UMAP reduz dimensionalidade dos embeddings
3. HDBSCAN descobre clusters (tópicos). Com consenso (padrão), UMAP + HDBSCAN rodam com várias sementes e
   os tópicos são os grupos que se repetem (consensus.py); o BERTopic recebe esses tópicos prontos
4. reduce_topics() une tópicos parecidos, se pedido (nr_topics)
5. reduce_outliers() leva o ruído para o tópico mais próximo, se configurado
6. CountVectorizer com stopwords de domínio, sem palavras presentes em quase todos os discursos e só com
   palavras de conteúdo (vocabulary.py) → c-TF-IDF → labels
7. KeyBERT + MMR refinam as labels pra diversidade e relevância
8. Multi-label: com consenso, a fração das rodadas em que o discurso fica junto de cada outro tópico;
   sem consenso, o pertencimento suave do HDBSCAN
"""

import re
import time
import warnings

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.cluster import BaseCluster
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

from .config import STOPWORDS_DOMAIN
from .consensus import coassociation, consensus_labels, membership
from .model_config import (
    CONSENSUS_SEED, EMBEDDING_MODELS, KEYBERT_TOP_N, LABEL_MAX_SEQ_LENGTH, LABEL_WORDS, MIN_DF, MMR_DIVERSITY,
    NGRAM_RANGE, TopicModelConfig, embedding_cache_key, methodology,
)
from .vocabulary import TOKEN_PATTERN, non_content_words, ubiquitous_words

warnings.filterwarnings('ignore')


# Verbo com pronome oblíquo ("criticá-la", "tornou-se", "fazê-lo-ia"): o separador de palavras corta
# no hífen e sobraria o verbo truncado ("criticá"); a forma inteira sai do vocabulário (é verbo).
_ENCLITIC = re.compile(
    r'\b[^\W\d_]+-(?:se|lhes?|l?[oa]s?|n[oa]s?|me|te|nos|vos)(?:-[^\W\d_]+)?\b', re.IGNORECASE,
)


def split_words(text: str, n_words: int | None) -> list[str]:
    """Divide o texto em trechos de n_words palavras; um resto curto (< 1/4) vai para o trecho anterior."""
    words = text.split()
    if not n_words or len(words) <= n_words:
        return [text]
    parts = [words[i:i + n_words] for i in range(0, len(words), n_words)]
    if len(parts[-1]) < n_words // 4:
        last = parts.pop()
        parts[-1] += last
    return [' '.join(p) for p in parts]


def load_embedder(model_key: str) -> SentenceTransformer:
    spec = EMBEDDING_MODELS[model_key]
    model = SentenceTransformer(spec.hf_name)
    model.max_seq_length = spec.max_seq_length
    if spec.fp16_on_gpu and model.device.type == 'cuda':
        model.half()
    return model


def iter_encode(texts: list[str], model_key: str, docs_per_block: int = 64):
    """
    Gera embeddings em blocos de discursos, para quem chama poder salvar e mostrar progresso.

    Cada discurso é dividido em trechos (se o modelo pedir); o embedding do discurso é a média
    normalizada dos embeddings dos trechos.

    Yields:
        (índice inicial do bloco, array float32 do bloco)
    """
    spec = EMBEDDING_MODELS[model_key]
    embedder = load_embedder(model_key)
    for start in range(0, len(texts), docs_per_block):
        block = texts[start:start + docs_per_block]
        pieces, owner = [], []
        for i, text in enumerate(block):
            for piece in split_words(text, spec.chunk_words):
                pieces.append(piece)
                owner.append(i)
        vecs = embedder.encode(pieces, batch_size=spec.batch_size, normalize_embeddings=True,
                               show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        owner = np.array(owner)
        out = np.vstack([vecs[owner == i].mean(axis=0) for i in range(len(block))])
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        yield start, out.astype(np.float32)


def device_info() -> dict:
    """{'device': 'cuda'|'cpu', 'gpu': nome da GPU ou None}."""
    import torch
    if torch.cuda.is_available():
        return {'device': 'cuda', 'gpu': torch.cuda.get_device_name(0)}
    return {'device': 'cpu', 'gpu': None}


class _ContentVectorizer(CountVectorizer):
    """CountVectorizer que descarta verbos com pronome oblíquo ("criticá-la") e bigramas de palavra
    repetida ("democracia democracia").

    Os bigramas repetidos surgem quando as stopwords entre duas ocorrências da mesma palavra são removidas.
    """

    def build_preprocessor(self):
        preprocess = super().build_preprocessor()
        return lambda doc: _ENCLITIC.sub(' ', preprocess(doc))

    def build_analyzer(self):
        analyze = super().build_analyzer()

        def analyzer(doc):
            out = []
            for term in analyze(doc):
                parts = term.split(' ')
                if len(parts) == 2 and parts[0] == parts[1]:
                    continue
                out.append(term)
            return out

        return analyzer


class TopicModeler:
    """Encapsula o pipeline de topic modeling."""

    def __init__(self, config: TopicModelConfig | None = None):
        self.config = config or TopicModelConfig()
        self.model: BERTopic | None = None
        self.embeddings: np.ndarray | None = None
        self.n_docs: int | None = None
        self.min_cluster_size: int | None = None

    def fit(
        self,
        documents: pd.Series,
        progress_callback=None,
        embeddings: np.ndarray | None = None,
    ) -> dict:
        """
        Roda o pipeline completo de topic modeling.

        Args:
            documents: Series com o texto limpo de cada discurso.
            progress_callback: Callable(stage: str, pct: float) opcional pra barra de progresso.
            embeddings: embeddings já calculados (mesma ordem de documents); se None, calcula.

        Returns:
            dict com:
                - topics: array de tópico principal por documento
                - reassigned: array bool — discurso era ruído e foi levado ao tópico mais próximo
                - n_noise: discursos que o HDBSCAN deixou como ruído (antes da reatribuição)
                - probabilities: matriz documentos × tópicos usada no multi-label: pertencimento suave do
                  HDBSCAN ou, com consenso, fração das rodadas junto de cada tópico (colunas = prob_topic_ids)
                - prob_topic_ids: tópico de cada coluna de probabilities
                - certainty: fração das rodadas do consenso em que o discurso ficou com o próprio tópico
                  (None sem consenso)
                - topic_labels: dict {topic_id: "label descritiva"}
                - multi_labels: list de listas — tópicos atribuídos a cada doc (multi-label)
                - model: o objeto BERTopic treinado
        """
        cfg = self.config
        _cb = progress_callback or (lambda *a: None)
        docs_list = documents.tolist()
        self.n_docs = len(docs_list)

        # ── 1. Embeddings ──
        if embeddings is None:
            _cb('embeddings', 0.0)
            embeddings = self._get_or_compute_embeddings(docs_list)
            _cb('embeddings', 1.0)

        # KeyBERTInspired needs the embedder even when embeddings are precomputed.
        # It only embeds candidate words and a few representative docs to pick labels, so a short
        # context is enough; long-context models (bge-m3: 8192 tokens) would otherwise exhaust GPU memory.
        embedder = load_embedder(cfg.embedding_model)
        embedder.max_seq_length = min(embedder.max_seq_length, LABEL_MAX_SEQ_LENGTH)

        self.min_cluster_size = cfg.resolved_min_cluster_size(len(docs_list))

        # ── 2. Tópicos de consenso (várias sementes) ou uma rodada de UMAP → HDBSCAN dentro do BERTopic ──
        coassoc = consensus_topics = None
        if cfg.consensus:
            _cb('consensus', 0.0)
            coassoc = coassociation(embeddings, cfg, self.min_cluster_size,
                                    progress=lambda pct: _cb('consensus', pct))
            consensus_topics = consensus_labels(coassoc, self.min_cluster_size)
            # BERTopic recebe os tópicos prontos (y) e só calcula as representações
            umap_model = BaseDimensionalityReduction()
            hdbscan_model = BaseCluster()
        else:
            umap_model = UMAP(
                n_neighbors=cfg.umap_n_neighbors,
                n_components=cfg.umap_n_components,
                min_dist=cfg.umap_min_dist,
                metric='cosine',
                random_state=CONSENSUS_SEED,
            )
            hdbscan_model = HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=cfg.min_samples,
                metric='euclidean',
                cluster_selection_method=cfg.spec.cluster_selection_method,
                prediction_data=True,
            )
        _cb('clustering', 0.0)

        vectorizer = _ContentVectorizer(
            stop_words=sorted(
                STOPWORDS_DOMAIN | ubiquitous_words(docs_list, STOPWORDS_DOMAIN) | non_content_words(docs_list)
            ),
            token_pattern=TOKEN_PATTERN,
            min_df=MIN_DF,
            ngram_range=NGRAM_RANGE,
        )

        ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

        representation = [
            KeyBERTInspired(top_n_words=KEYBERT_TOP_N),
            MaximalMarginalRelevance(diversity=MMR_DIVERSITY),
        ]

        self.model = BERTopic(
            embedding_model=embedder,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer,
            ctfidf_model=ctfidf,
            representation_model=representation,
            language='portuguese',
            # pertencimento suave a cada tópico, usado no multi-label sem consenso
            calculate_probabilities=not cfg.consensus,
            verbose=True,
            min_topic_size=self.min_cluster_size,
        )

        # ── 3. Fit ──
        topics, probs = self.model.fit_transform(docs_list, embeddings, y=consensus_topics)

        # ── 4. Reduzir tópicos (se configurado) ──
        if cfg.nr_topics is not None:
            _cb('reduce', 0.0)
            self.model.reduce_topics(docs_list, nr_topics=cfg.nr_topics)
            topics = self.model.topics_
            probs = self.model.probabilities_

        # ── 5. Ruído → tópico mais próximo (se configurado) ──
        topics = np.array(topics)
        noise = topics == -1
        if cfg.reduce_outliers and noise.any() and (topics != -1).any():
            new_topics = self.model.reduce_outliers(
                docs_list, list(topics), strategy='embeddings', embeddings=embeddings,
            )
            self.model.update_topics(
                docs_list, topics=new_topics, vectorizer_model=vectorizer,
                ctfidf_model=ctfidf, representation_model=representation,
            )
            topics = np.array(new_topics)
        reassigned = noise & (topics != -1)

        # ── 6. Labels ──
        topic_labels = self._extract_labels()

        # ── 7. Certeza e multi-label ──
        certainty = None
        prob_ids = list(range(probs.shape[1])) if probs is not None and probs.ndim == 2 else []
        if coassoc is not None:
            # núcleo de cada tópico = discursos que o consenso agrupou (sem os reatribuídos)
            ids, member = membership(coassoc, topics, core=~noise)
            col = {t: j for j, t in enumerate(ids)}
            certainty = np.array([member[i, col[t]] if t in col else np.nan for i, t in enumerate(topics)])
            probs, prob_ids = member, ids
            del coassoc
        multi_labels = self._assign_multi_labels(probs, prob_ids, topics, topic_labels)

        self.embeddings = embeddings

        return {
            'topics': topics,
            'reassigned': reassigned,
            'n_noise': int(noise.sum()),
            'probabilities': probs,
            'prob_topic_ids': prob_ids,
            'certainty': certainty,
            'topic_labels': topic_labels,
            'multi_labels': multi_labels,
            'model': self.model,
        }

    def methodology(self) -> list[dict]:
        """Parâmetros usados em cada etapa, em formato de tabela (para o paper)."""
        return methodology(self.config, self.n_docs)

    def _get_or_compute_embeddings(self, docs: list[str]) -> np.ndarray:
        """Gera embeddings ou carrega do cache em disco (linha de comando)."""
        cfg = self.config
        cache_path = None

        if cfg.cache_dir:
            cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            model_slug = embedding_cache_key(cfg.embedding_model).replace('/', '_').replace('|', '_')
            cache_path = cfg.cache_dir / f'embeddings_{model_slug}_{len(docs)}.npy'
            if cache_path.exists():
                print(f"Carregando embeddings do cache: {cache_path}")
                return np.load(cache_path)

        print(f"Gerando embeddings com {cfg.spec.hf_name}...")
        t0 = time.time()
        embeddings = np.vstack([v for _, v in iter_encode(docs, cfg.embedding_model)])
        print(f"Embeddings prontos em {time.time() - t0:.0f}s")

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
                top_words = ' | '.join(w for w, _ in words[:LABEL_WORDS])
                labels[topic_id] = f"T{topic_id}: {top_words}"
        return labels

    def _assign_multi_labels(
        self,
        probs: np.ndarray | None,
        prob_ids: list[int],
        topics: np.ndarray,
        labels: dict[int, str],
    ) -> list[list[str]]:
        """
        Tópicos de cada discurso: o principal primeiro, depois os outros cujo valor em probs
        (coluna j = tópico prob_ids[j]) passa do threshold, do maior para o menor.
        """
        threshold = self.config.multi_label_threshold
        multi = []
        for i, main in enumerate(topics):
            doc = [labels[main]] if main != -1 and main in labels else []
            if probs is not None and probs.ndim == 2:
                for j in np.argsort(-probs[i]):
                    if probs[i, j] < threshold:
                        break
                    t = prob_ids[j]
                    if t != main and t in labels:
                        doc.append(labels[t])
            multi.append(doc)
        return multi

    def get_topic_summary(self) -> pd.DataFrame:
        """Retorna resumo dos tópicos pra visualização."""
        if self.model is None:
            raise RuntimeError("Modelo não treinado. Chame fit() primeiro.")
        return self.model.get_topic_info()
