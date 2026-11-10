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
"""

from collections import Counter, defaultdict

SPACY_MODEL = 'pt_core_news_sm'
CONTENT_POS = {'NOUN', 'PROPN', 'ADJ', 'PART'}  # PART = particípio (VERB/AUX com VerbForm=Part)
MIN_CONTENT_SHARE = 0.5  # palavra fica se ≥50% das ocorrências forem de uma classe de conteúdo

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
