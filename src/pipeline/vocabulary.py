"""
Vocabulário dos rótulos dos tópicos: só palavras de conteúdo.

Os rótulos saem do c-TF-IDF, que pontua cada palavra pelo quanto ela é típica de um tópico.
Verbos e advérbios de fala ("falando", "devemos", "infelizmente", "inclusive") são típicos de
discursos de qualquer tema e acabavam nos rótulos. Listar um por um não acaba nunca, então
o vocabulário é filtrado por classe gramatical (spaCy):

- ficam substantivos, nomes próprios, adjetivos e particípios ("assassinada", "condenado",
  "sancionada"), que carregam o tema;
- saem verbos (exceto particípios), advérbios, pronomes, numerais, etc.

A classe de uma palavra é a mais comum dela no corpus analisado: "fala" aparece mais como
verbo ("ela fala") do que como substantivo ("a fala"), então sai.

Palavras presentes na maioria dos discursos também saem (ubiquitous_words): aparecem em quase
todos os tópicos e não diferenciam nenhum. Em geral são a própria keyword de uma busca com um termo
só e o vocabulário que a acompanha; numa busca com várias keywords, cada uma aparece em parte dos
discursos e continua podendo aparecer nos rótulos.

O mesmo vocabulário (excluded_words + ContentVectorizer) é usado nas palavras características de cada
espectro (statistics.characteristic_words).
"""

import re
import unicodedata
from collections import Counter, defaultdict

from sklearn.feature_extraction.text import CountVectorizer

from .config import STOPWORDS_DOMAIN

SPACY_MODEL = 'pt_core_news_sm'
CONTENT_POS = {'NOUN', 'PROPN', 'ADJ', 'PART'}  # PART = particípio (VERB/AUX com VerbForm=Part)
MIN_CONTENT_SHARE = 0.5  # palavra fica se ≥50% das ocorrências forem de uma classe de conteúdo

TOKEN_PATTERN = r'(?u)\b[^\W\d_]{2,}\b'  # só palavras com letras (sem números como "2023", "16")
# palavra em ≥50% dos discursos sai do vocabulário. Na coleta "feminicídio" (nov/2026), 90% tirava só
# "feminicídio" e os rótulos se enchiam de "violência mulher" (72% e 84% dos discursos); entre 90% e 50%
# só saem essas; "crime" e "vida" (40%) ficam.
UBIQUITOUS_SHARE = 0.5

# Verbo com pronome oblíquo ("criticá-la", "tornou-se", "fazê-lo-ia"): o separador de palavras corta
# no hífen e sobraria o verbo truncado ("criticá"); a forma inteira sai do vocabulário (é verbo).
_ENCLITIC = re.compile(
    r'\b[^\W\d_]+-(?:se|lhes?|l?[oa]s?|n[oa]s?|me|te|nos|vos)(?:-[^\W\d_]+)?\b', re.IGNORECASE,
)


class ContentVectorizer(CountVectorizer):
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


def fold(text: str) -> str:
    """Minúsculas e sem acento ("Feminicídio" → "feminicidio")."""
    nfkd = unicodedata.normalize('NFKD', text.lower())
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def _variant_key(word: str) -> str:
    """Mesma chave para formas com/sem acento e singular/plural em -s ("feminicídios" → "feminicidio")."""
    w = fold(word)
    return w[:-1] if w.endswith('s') and len(w) > 3 else w


def ubiquitous_words(documents: list[str], ignore: set[str] = frozenset(),
                     share: float = UBIQUITOUS_SHARE) -> set[str]:
    """
    Palavras (minúsculas) presentes em ≥ share dos discursos, contando juntas as variações de acento
    e plural em -s: com "feminicídio" ou "feminicídios" na maioria dos discursos, saem
    "feminicídio", "feminicidio", "feminicídios"...

    ignore: palavras que já são stopwords; ficam fora da contagem para não arrastarem palavras de
    conteúdo que só diferem pelo acento ("para" → "pará").
    """
    if not documents:
        return set()
    token = re.compile(TOKEN_PATTERN)
    df = Counter()                       # discursos por variante
    forms = defaultdict(set)             # variante → formas vistas no corpus
    for doc in documents:
        words = set(token.findall(doc.lower())) - ignore
        keys = set()
        for w in words:
            k = _variant_key(w)
            forms[k].add(w)
            keys.add(k)
        df.update(keys)
    return {w for k, n in df.items() if n / len(documents) >= share for w in forms[k]}

_nlp = None


def _load_spacy():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load(SPACY_MODEL, disable=['parser', 'ner', 'lemmatizer'])
        except OSError as e:
            raise RuntimeError(
                f"Modelo do spaCy '{SPACY_MODEL}' não instalado. "
                f"Rode: python -m spacy download {SPACY_MODEL}"
            ) from e
    return _nlp


def non_content_words(documents: list[str]) -> set[str]:
    """Palavras (minúsculas) cuja classe mais comum no corpus não é de conteúdo."""
    nlp = _load_spacy()
    pos: dict[str, Counter] = defaultdict(Counter)
    for doc in nlp.pipe(documents, batch_size=32):
        for tok in doc:
            if not tok.is_alpha:
                continue
            tag = tok.pos_
            if tag in ('VERB', 'AUX') and 'Part' in tok.morph.get('VerbForm'):
                tag = 'PART'
            pos[tok.text.lower()][tag] += 1

    return {
        word for word, counts in pos.items()
        if sum(counts[p] for p in CONTENT_POS) / sum(counts.values()) < MIN_CONTENT_SHARE
    }


def excluded_words(documents: list[str]) -> set[str]:
    """Tudo o que fica fora do vocabulário: stopwords, palavras ubíquas e palavras que não são de conteúdo."""
    return STOPWORDS_DOMAIN | ubiquitous_words(documents, STOPWORDS_DOMAIN) | non_content_words(documents)
