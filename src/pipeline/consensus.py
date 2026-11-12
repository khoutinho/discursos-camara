"""
Tópicos de consenso: o agrupamento repetido com várias sementes do UMAP.

Uma rodada só de UMAP → HDBSCAN depende da semente: com outra, parte dos discursos muda de
tópico (ARI ~0,6 entre sementes em "feminicídio", bge-m3). Aqui o agrupamento roda N vezes e
conta, para cada par de discursos, a fração das rodadas em que os dois ficaram no mesmo grupo
(matriz de coassociação). Os tópicos finais são os grupos de discursos que ficam juntos em mais
da metade das rodadas (ligação média, corte em 1 − CONSENSUS_CUT).

Testado em nov/2026 ("feminicídio" 2015–2026, bge-m3): dois consensos feitos com sementes
independentes concordam com ARI 0,88, contra 0,60–0,68 entre rodadas únicas; 15 rodadas dão o
mesmo resultado que 30. HDBSCAN sobre a coassociação não melhorava (ARI 0,69).

A mesma matriz dá, por discurso:
- certeza: fração das rodadas em que ele fica com os discursos do próprio tópico;
- multi-label: fração das rodadas em que fica com os de cada outro tópico. Abaixo de 0,2 é quase
  sempre acaso (com outras sementes, só 25% desses 2º tópicos se repetiam; acima de 0,2, 98%).
"""

import numpy as np
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from umap import UMAP

from .model_config import CONSENSUS_CUT, CONSENSUS_SEED


def coassociation(embeddings: np.ndarray, cfg, min_cluster_size: int, progress=None) -> np.ndarray:
    """
    Matriz n × n (float32): fração das cfg.consensus_runs rodadas em que cada par ficou no mesmo grupo
    do HDBSCAN. Ruído não conta como "junto". A diagonal é a fração das rodadas em que o discurso foi
    agrupado. Memória: 4 bytes × n² (10.000 discursos → 400 MB).
    """
    n = len(embeddings)
    runs = cfg.consensus_runs
    C = np.zeros((n, n), dtype=np.float32)
    for r in range(runs):
        reduced = UMAP(
            n_neighbors=cfg.umap_n_neighbors, n_components=cfg.umap_n_components, min_dist=cfg.umap_min_dist,
            metric='cosine', random_state=CONSENSUS_SEED + r,
        ).fit_transform(embeddings)
        labels = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=cfg.min_samples, metric='euclidean',
            cluster_selection_method=cfg.spec.cluster_selection_method,
        ).fit_predict(reduced)
        for t in np.unique(labels):
            if t == -1:
                continue
            idx = np.flatnonzero(labels == t)
            C[np.ix_(idx, idx)] += 1
        if progress:
            progress((r + 1) / runs)
    C /= runs
    return C


def consensus_labels(C: np.ndarray, min_cluster_size: int, cut: float = CONSENSUS_CUT) -> np.ndarray:
    """
    Tópicos de consenso: ligação média sobre 1 − C, cortada em 1 − cut. Grupos com menos de
    min_cluster_size discursos ficam sem tópico (-1), como o ruído do HDBSCAN. Ids por tamanho (0 = maior).
    """
    D = 1.0 - C.astype(np.float64)
    np.fill_diagonal(D, 0.0)
    groups = fcluster(linkage(squareform(D, checks=False), method='average'), t=1.0 - cut, criterion='distance')
    ids, sizes = np.unique(groups, return_counts=True)
    keep = [g for g, s in sorted(zip(ids, sizes), key=lambda x: -x[1]) if s >= min_cluster_size]
    new_id = {g: i for i, g in enumerate(keep)}
    return np.array([new_id.get(g, -1) for g in groups])


def membership(C: np.ndarray, topics: np.ndarray, core: np.ndarray) -> tuple[list[int], np.ndarray]:
    """
    (ids dos tópicos, matriz n × tópicos): fração média das rodadas em que cada discurso ficou junto
    dos discursos do núcleo de cada tópico (os que o consenso agrupou; sem os reatribuídos), sem contar
    o próprio discurso.
    """
    ids = sorted(t for t in np.unique(topics) if t != -1)
    diag = np.diag(C)
    M = np.zeros((len(C), len(ids)), dtype=np.float32)
    for j, t in enumerate(ids):
        members = (topics == t) & core
        if not members.any():  # tópico sem núcleo (não acontece com o consenso); usa todos os membros
            members = topics == t
        total = C[:, members].sum(axis=1) - np.where(members, diag, 0.0)
        count = members.sum() - members.astype(int)
        M[:, j] = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    return ids, M
