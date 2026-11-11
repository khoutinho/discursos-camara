"""
Gera acentos_pt_BR.tsv: mapa "palavra sem acento → formas com acento".

Usado pelo formulário de coleta para sugerir a forma acentuada de uma keyword
(ver accents.py). Só o .tsv gerado vai para o repositório; este script só precisa
rodar de novo para atualizar o dicionário.

Fontes:
- Dicionário pt_BR do LibreOffice (projeto VERO, Raimundo Moura e equipe),
  licença LGPLv3 ou MPL 2.0. As palavras-base são expandidas pelas regras de
  afixo do Hunspell (plurais, femininos, conjugações).
- Lista de frequência do wordfreq (Robyn Speer), dados CC BY-SA 4.0. Só entram
  formas que existem no wordfreq: a expansão gera lixo de verbo + pronome
  ("camará", de "camar-á") e milhões de conjugações raras.

Rodar (dependências só deste script):
    pip install spylls wordfreq
    python -m src.scraper.build_accents
"""

import sys
import tempfile
import unicodedata
from pathlib import Path

import requests

VERO_COMMIT = "1e848fbddd7fd8e03fb696ecc03ee1068fab141c"
VERO_URL = f"https://raw.githubusercontent.com/LibreOffice/dictionaries/{VERO_COMMIT}/pt_BR/pt_BR"
OUTPUT = Path(__file__).parent / "acentos_pt_BR.tsv"

LETTERS = set("abcdefghijklmnopqrstuvwxyzáàâãéêíóôõúüç")

# Formas acentuadas com menos de 1% da frequência da mais comum saem ("aguá" ao lado de "água").
MIN_ACCENTED_RATIO = 0.01
# A forma sem acento entra como opção quando é palavra do dicionário e aparece com pelo menos
# 3% da frequência da forma acentuada mais comum. Abaixo disso a ocorrência é quase toda erro de
# digitação: "politica" tem 2,6% de "política"; "publico" (de publicar) tem 3,4% e "pais", 41%.
MIN_BARE_RATIO = 0.03


def strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def is_enclitic_stem(sfx) -> bool:
    """
    Regra que tira o "r" do infinitivo e acentua a vogal: votar → "votá", para "votá-lo".

    O Hunspell aceita "votá" sozinho porque separa a palavra no hífen, e o wordfreq também conta
    "votá" de "votá-lo". A mesma regra gera formas de verdade ("está", "dá", "vê").
    """
    return (
        sfx.strip.endswith("r")
        and len(sfx.add) == len(sfx.strip) - 1
        and sfx.add != sfx.strip[:-1]
        and strip_accents(sfx.add) == sfx.strip[:-1]
    )


def expand(dictionary) -> tuple[set[str], set[str]]:
    """
    Todas as formas geradas pelas regras de afixo (sufixos encadeados até 2 níveis).

    Devolve (formas, formas que só saem de is_enclitic_stem).
    """
    aff = dictionary.aff

    def suffixed(word, flags):
        for flag in flags:
            for sfx in aff.SFX.get(flag, []):
                if word.endswith(sfx.strip) and sfx.cond_regexp.search(word):
                    yield word[: len(word) - len(sfx.strip)] + sfx.add, sfx

    def prefixed(word, flags):
        for flag in flags:
            for pfx in aff.PFX.get(flag, []):
                if word.startswith(pfx.strip) and pfx.cond_regexp.search(word):
                    yield pfx.add + word[len(pfx.strip):], pfx

    forms, enclitic = set(), set()
    for entry in dictionary.dic.words:
        flags = entry.flags
        if aff.FORBIDDENWORD and aff.FORBIDDENWORD in flags:
            continue
        if not (aff.NEEDAFFIX and aff.NEEDAFFIX in flags):
            forms.add(entry.stem)
        for form, sfx in suffixed(entry.stem, flags):
            (enclitic if is_enclitic_stem(sfx) else forms).add(form)
            forms.update(f for f, _ in suffixed(form, sfx.flags))
            if sfx.crossproduct:
                forms.update(f for f, pfx in prefixed(form, flags) if pfx.crossproduct)
        forms.update(f for f, _ in prefixed(entry.stem, flags))

    def clean(words):
        return {w.lower() for w in words if set(w.lower()) <= LETTERS}

    forms = clean(forms)
    enclitic = clean(enclitic) - forms
    return forms | enclitic, enclitic


def build_map(forms: set[str], enclitic: set[str], freq: dict[str, float]) -> dict[str, list[str]]:
    accented: dict[str, list[str]] = {}
    for form in forms:
        bare = strip_accents(form)
        if bare != form and form in freq:
            accented.setdefault(bare, []).append(form)

    result = {}
    for bare, options in accented.items():
        # Forma de is_enclitic_stem só fica se for a mais comum de todas: "está" (mais que "esta")
        # fica; "votá" (menos que "vota") e "abraçá" (menos que "abraça") saem.
        rivals = [freq[f] for f in options if f not in enclitic] + [freq.get(bare, 0)]
        options = [f for f in options if f not in enclitic or freq[f] >= max(rivals)]
        if not options:
            continue
        top = max(freq[f] for f in options)
        bare_is_word = bare in forms and freq.get(bare, 0) >= top * MIN_BARE_RATIO
        # Quando a forma sem acento é a mais comum ("para", "uma"), ela também conta como referência:
        # "pará" e "umã" ficam abaixo de 1% e saem.
        ref = max(top, freq.get(bare, 0)) if bare_is_word else top
        options = sorted((f for f in options if freq[f] >= ref * MIN_ACCENTED_RATIO), key=lambda f: -freq[f])
        if not options:
            continue
        result[bare] = options + [bare] if bare_is_word else options
    return dict(sorted(result.items()))


def main():
    from spylls.hunspell import Dictionary
    from wordfreq import get_frequency_dict

    with tempfile.TemporaryDirectory() as tmp:
        for ext in ("dic", "aff"):
            resp = requests.get(f"{VERO_URL}.{ext}", timeout=60)
            resp.raise_for_status()
            (Path(tmp) / f"pt_BR.{ext}").write_bytes(resp.content)
        dictionary = Dictionary.from_files(str(Path(tmp) / "pt_BR"))

    forms, enclitic = expand(dictionary)
    mapping = build_map(forms, enclitic, get_frequency_dict("pt", wordlist="large"))

    header = [
        "# Palavra sem acento → formas com acento, da mais para a menos frequente.",
        "# Quando a forma sem acento também é palavra (pais/país), ela aparece por último.",
        "# Gerado por src/scraper/build_accents.py.",
        f"# Dicionário pt_BR do LibreOffice (projeto VERO), commit {VERO_COMMIT}, LGPLv3/MPL 2.0.",
        "# Filtrado pela lista de frequência do wordfreq (Robyn Speer), CC BY-SA 4.0.",
        "# Este arquivo é distribuído sob CC BY-SA 4.0.",
    ]
    lines = [f"{bare}\t{' '.join(options)}" for bare, options in mapping.items()]
    OUTPUT.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print(f"{len(forms):,} formas no dicionário → {len(mapping):,} entradas em {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
