"""
Formas acentuadas de palavras digitadas sem acento.

O índice do SitaQWeb mudou no fim de 2022: discursos até 2022 só são encontrados pela forma
acentuada da keyword. _search_expression (sitaqweb.py) tira o acento mas não consegue colocar,
então o formulário de coleta usa este mapa para sugerir a forma acentuada.
Mapa gerado por build_accents.py.
"""

from functools import cache
from pathlib import Path

MAP_FILE = Path(__file__).parent / "acentos_pt_BR.tsv"


@cache
def _load() -> dict[str, list[str]]:
    mapping = {}
    with MAP_FILE.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            bare, options = line.rstrip("\n").split("\t")
            mapping[bare] = options.split(" ")
    return mapping


def accented_forms(word: str) -> list[str]:
    """
    Opções para uma palavra sem acento, da mais para a menos frequente.

    [] se a palavra já tem acento ou não tem forma acentuada. Quando a forma sem acento também
    é palavra ("pais"), ela vem por último: accented_forms("pais") == ["país", "pais"].
    """
    return list(_load().get(word.lower(), []))
