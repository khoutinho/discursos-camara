"""
Limpeza do texto do discurso antes do topic modeling.

O texto das notas taquigráficas traz, além da fala, partes que não dizem nada sobre o tema
e que fazem discursos diferentes parecerem iguais para o modelo de embedding:

- cabeçalho do orador: "O SR. FULANO (PT - SP. Sem revisão do orador.) - "
- marcações de discurso entregue por escrito: "DISCURSO NA ÍNTEGRA ENCAMINHADO PELO SR. DEPUTADO ..."
- falas de outras pessoas no meio: Presidente da sessão ("Conclua, Deputada"), apartes
- marcações da sessão: "(Palmas.)", "(Desligamento automático do microfone.)"
- vocativos e cumprimentos de abertura: "Sr. Presidente, Sras. e Srs. Deputados, ..."
- fechamentos: "Era o que tinha a dizer.", pedido de divulgação na Voz do Brasil

O texto original continua guardado; esta limpeza só define o que entra no modelo.
Mudou alguma regra? Incremente CLEANING_VERSION para invalidar o cache de embeddings.
"""

import re
import unicodedata

CLEANING_VERSION = 1

# ── Marcadores de troca de orador ──
# "O SR. NOME (Partido - UF. ...) - " ou "A SRA. PRESIDENTE (Fulana. PL - SP) - "
_UPPER = r"A-ZÀ-ÖØ-Þ"
_SPEAKER = re.compile(
    rf"(?:^|(?<=[\s.!?:)\]…]))(?:O|A)\s+SRA?\.\s+([{_UPPER}][{_UPPER}'’.\- ]*?)\s*"
    r"(?:\([^()]*\))?\s*[-–—]\s+"
)
# "DISCURSO NA ÍNTEGRA ENCAMINHADO PELO SR. DEPUTADO NOME (SEM REGISTRO TAQUIGRÁFICO)."
# e "DOCUMENTO ENCAMINHADO PELA SRA. DEPUTADA NOME" (anexo do orador)
_FULL_TEXT = re.compile(
    r"(?:DISCURSO|DOCUMENTO|PARECER\s+ESCRITO)S?\s+(?:NA\s+ÍNTEGRA\s+)?ENCAMINHADOS?\s+(?:À\s+MESA\s+)?"
    r"PEL[OA]\s+SRA?\.\s+DEPUTAD[OA]\s+"
    rf"((?:[{_UPPER}][{_UPPER}'’\-]*\.?[ ]?)+?)"           # nome: palavras em caixa alta
    rf"(?=\s*\(|\s*[.:]\s|\s*$|\s+(?![{_UPPER}]+\b))"     # termina antes de "(", ".", ou texto normal
    r"\s*(?:\(SEM REGISTRO TAQUIGRÁFICO\))?\s*[.:]?\s*"
)
# O orador volta a falar (por escrito) ou anexa um documento:
# "PRONUNCIAMENTO ENCAMINHADO PELO ORADOR", "DOCUMENTO A QUE SE REFERE A ORADORA"
_BACK_TO_SPEAKER = re.compile(
    r"PRONUNCIAMENTOS?\s+ENCAMINHADOS?\s+PEL[OA]\s+ORADORA?\s*\.?\s*"
    rf"|(?:[{_UPPER}]+\s+){{1,3}}A\s+QUE\s+SE\s+REFERE\s+(?:O|A)\s+ORADORA?\s*\.?\s*"
)
_TO_DROP = [
    re.compile(r"PRONUNCIAMENTOS?\s+ENCAMINHADOS?\s+À\s+MESA\s+PARA\s+PUBLICAÇÃO\s*"),
]

# ── Marcações da sessão entre parênteses ──
_STAGE = re.compile(
    r"\(\s*(?:palmas|pausa|risos|soa a campainha|desligamento[^)]*|interrupção[^)]*|"
    r"manifestaç[^)]*|falha na transmissão|apupos|vaias|tumulto[^)]*|\.\.\.|…)\s*\.?\s*\)",
    re.IGNORECASE,
)

# ── Abertura: vocativos e cumprimentos ──
# Abreviações cujo ponto não termina frase (trocado por um marcador durante a limpeza)
_ABBREV = re.compile(r"\b(Sra?s?|Srªs?|Exma?s?|Dra?s?|Profa?s?|V\.Exa?s?|S\.Exa?s?|V\.Sa?s?|n|Art|arts?)\.", re.IGNORECASE)
_DOT = "․"  # "one dot leader", invisível para o leitor e não usado como fim de frase aqui

# Vocativo: o trecho COMEÇA (depois de artigos/adjetivos de cortesia) com um destes destinatários
_VOCATIVE = re.compile(
    r"^(?:(?:e|a|o|as|os|aos?|às?|meus?|minhas?|nossos?|nossas?|car[oa]s?|prezad[oa]s?|querid[oa]s?|"
    r"nobres?|demais|ilustres?|digníssim[oa]s?|excelentíssim[oa]s?|exm[oa]s?|senhoras?\s+e)\.?\s+)*"
    r"(?:presidente|presidenta|deputad[oa]s?|parlamentares?|senador(?:a|es|as)?|congressistas?|colegas?|"
    r"pares|senhor(?:a|es|as)?|sra?s?|srªs?|excelência|tod[oa]s|presentes|convidad[oa]s?|ouvintes?|"
    r"telespectador(?:a|es|as)?|internautas?|mesa|brasileir[oa]s?|ministr[oa]s?|líder(?:es)?)\b",
    re.IGNORECASE,
)
_GREETING_START = re.compile(
    r"^(?:muit[oa]\s+)?(?:obrigad[oa]|agradeç|boa\s+(?:tarde|noite)|bom\s+dia|cumpriment|saúdo|"
    r"olá|gente\b)",
    re.IGNORECASE,
)
_MAX_VOCATIVE_WORDS = 12
_MAX_GREETING_WORDS = 25

# ── Fechamento ──
_CLOSING = re.compile(r"Era\s+o\s+que\s+(?:eu\s+)?tinha\s+(?:a|para)\s+dizer[^.!?]*[.!?]?", re.IGNORECASE)
_PUBLICITY_VERB = re.compile(r"divulga|divulgu|veicul|transmiti|publica|registr|consign|repercuss", re.IGNORECASE)
_PUBLICITY_PLACE = re.compile(
    r"voz do brasil|(?:meios|órgãos|canais|veículos)\s+de\s+comunicação|comunicação\s+(?:da|desta)\s+(?:casa|câmara)|"
    r"\banais\b|programa\s+a\s+voz",
    re.IGNORECASE,
)


def _norm_name(name: str) -> str:
    """Nome sem acentos, maiúsculo e com espaços simples, para comparar orador e cabeçalho."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", s).strip(" .").upper()


def _same_person(a: str, b: str) -> bool:
    return bool(a and b) and (a == b or a in b or b in a)


def _split_turns(text: str) -> list[tuple[str | None, str]]:
    """Divide o texto em falas: [(orador ou None se antes de qualquer marcador, texto)]."""
    markers = []
    for m in _SPEAKER.finditer(text):
        markers.append((m.start(), m.end(), m.group(1)))
    for m in _FULL_TEXT.finditer(text):
        markers.append((m.start(), m.end(), m.group(1)))
    for m in _BACK_TO_SPEAKER.finditer(text):
        markers.append((m.start(), m.end(), "")) # "" = o orador principal
    markers.sort()

    turns: list[tuple[str | None, str]] = []
    pos, speaker = 0, None
    for start, end, who in markers:
        if start < pos:  # marcador sobreposto a outro já consumido
            continue
        turns.append((speaker, text[pos:start]))
        pos, speaker = end, who
    turns.append((speaker, text[pos:]))
    return [(s, t) for s, t in turns if t.strip()]


def _main_speaker_text(text: str, orador: str | None) -> str:
    """Mantém só as falas do orador principal (descarta Presidente da sessão e apartes)."""
    turns = _split_turns(text)
    if not turns:
        return ""
    named = [s for s, _ in turns if s]
    main = _norm_name(orador) if orador else ""
    if not main or not any(_same_person(main, _norm_name(s)) for s in named):
        main = _norm_name(named[0]) if named else ""

    kept = [t for s, t in turns if s is None or s == "" or (main and _same_person(main, _norm_name(s)))]
    if not kept:  # não deu para identificar o orador: fica tudo, menos a Presidência
        kept = [t for s, t in turns if not (s and _norm_name(s).startswith("PRESIDENT"))] or [text]
    return " ".join(kept)


def _strip_opening(text: str) -> str:
    """Remove vocativos e cumprimentos do começo da fala."""
    while True:
        text = text.lstrip(" ,.;:-–—!?\n\t")
        m = re.match(r"([^,.;!?:]*)([,.;!?:]+|$)", text)
        if not m or not m.group(1).strip():
            return text
        chunk = m.group(1).strip()
        n_words = len(chunk.split())
        if _GREETING_START.match(chunk):
            # cumprimento curto: remove a frase inteira
            sent = re.match(r"[^.!?]*[.!?]?", text).group(0)
            if len(sent.split()) <= _MAX_GREETING_WORDS:
                text = text[len(sent):]
                continue
            return text
        if n_words <= _MAX_VOCATIVE_WORDS and _VOCATIVE.match(chunk) and not re.search(
            r"\b(?:quero|venho|subo|gostaria|vou|peço|solicito|é|são|foi|está|estão|tem|têm|há)\b", chunk, re.IGNORECASE
        ):
            text = text[m.end():]
            continue
        return text


def _drop_closing(text: str) -> str:
    """Remove "Era o que tinha a dizer" e pedidos de divulgação."""
    text = _CLOSING.sub(" ", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sentences if not (_PUBLICITY_VERB.search(s) and _PUBLICITY_PLACE.search(s))]
    # agradecimentos curtos no final
    while kept and _GREETING_START.match(kept[-1].strip()) and len(kept[-1].split()) <= 8:
        kept.pop()
    return " ".join(kept)


def clean_speech(text: str | None, orador: str | None = None) -> str:
    """
    Texto do discurso pronto para o embedding e o c-TF-IDF.

    Args:
        text: texto original do discurso.
        orador: nome do orador (coluna Orador), usado para separar a fala dele das falas de outros.
            Sem ele, o orador principal é o do primeiro cabeçalho.

    Returns:
        o texto limpo; se a limpeza apagar tudo, o texto original sem o cabeçalho.
    """
    if not text or not isinstance(text, str):
        return ""
    original = text
    for pat in _TO_DROP:
        text = pat.sub(" ", text)
    text = _main_speaker_text(text, orador)
    text = _STAGE.sub(" ", text)

    text = _ABBREV.sub(lambda m: m.group(0)[:-1] + _DOT, text)
    text = _strip_opening(text)
    text = _drop_closing(text)
    text = text.replace(_DOT, ".")
    text = re.sub(r"\s+", " ", text).strip()

    if len(text.split()) < 5:  # sobrou quase nada: melhor o original sem o cabeçalho
        text = re.sub(r"\s+", " ", _SPEAKER.sub(" ", original, count=1)).strip()
    return text


MODEL_TEXT_COLUMN = 'Texto_modelo'


def add_model_text(df):
    """Adiciona a coluna Texto_modelo (texto limpo) ao DataFrame, se ainda não existir."""
    if MODEL_TEXT_COLUMN in df.columns:
        return df
    df = df.copy()
    oradores = df['Orador'] if 'Orador' in df.columns else [None] * len(df)
    df[MODEL_TEXT_COLUMN] = [clean_speech(t, o) for t, o in zip(df['Discurso'], oradores)]
    return df
