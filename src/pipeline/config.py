"""
Constantes do pipeline: classificação ideológica, regiões, keywords e stopwords.

Classificação ideológica: Bolognesi, Ribeiro & Codato (2023)
Escala 0-10: extrema-esquerda(0-1.5) até extrema-direita(8.51-10)
"""

import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

# ─── Classificação ideológica (7 classes) ────────────────────────────────────

ESPECTRO = {
    # Extrema-esquerda
    'PSOL': 'Extrema-esquerda', 'PCdoB': 'Extrema-esquerda',
    'PCB': 'Extrema-esquerda', 'UP': 'Extrema-esquerda',

    # Esquerda
    'PT': 'Esquerda', 'PDT': 'Esquerda', 'PSB': 'Esquerda',
    'REDE': 'Esquerda', 'PV': 'Esquerda', 'PMN': 'Esquerda',

    # Centro-esquerda
    'PPS': 'Centro-esquerda', 'PROS': 'Centro-esquerda',

    # Centro
    'MDB': 'Centro', 'PSDB': 'Centro', 'CIDADANIA': 'Centro',
    'PMDB': 'Centro', 'AVANTE': 'Centro',

    # Centro-direita
    'PSD': 'Centro-direita', 'PODE': 'Centro-direita',
    'SOLIDARIEDADE': 'Centro-direita', 'PRD': 'Centro-direita',

    # Direita
    'PL': 'Direita', 'PP': 'Direita', 'PROGRESSISTAS': 'Direita',
    'REPUBLICANOS': 'Direita', 'UNIAO': 'Direita', 'UNIÃO': 'Direita',
    'DEM': 'Direita', 'PSC': 'Direita', 'PRB': 'Direita',
    'PTB': 'Direita', 'PR': 'Direita', 'PATRIOTA': 'Direita',
    'PFL': 'Direita',

    # Extrema-direita
    'PSL': 'Extrema-direita', 'NOVO': 'Extrema-direita',

    # Partidos extintos/históricos (presentes no CSV completo 2000-2025)
    'PTDOB': 'Centro', 'PTdoB': 'Centro',
    'PRONA': 'Extrema-direita',
    'PTC': 'Centro-direita', 'PRP': 'Direita',
    'SDD': 'Centro-direita',
    'PMB': 'Centro-direita',
    'PRTB': 'Extrema-direita',
    'PATRI': 'Direita',
    'PAN': 'Centro-esquerda',
    'PST': 'Centro-direita',
    'PHS': 'Centro-direita',
    'PPB': 'Direita', 'PPR': 'Direita',
    'PDS': 'Direita', 'ARENA': 'Direita',
    'PPL': 'Esquerda',
    'PCO': 'Extrema-esquerda', 'PSTU': 'Extrema-esquerda',
}

ESPECTRO_3 = {
    'Extrema-esquerda': 'Esquerda',
    'Esquerda': 'Esquerda',
    'Centro-esquerda': 'Esquerda',
    'Centro': 'Centro',
    'Centro-direita': 'Direita',
    'Direita': 'Direita',
    'Extrema-direita': 'Direita',
}

# ─── Classificação geográfica ────────────────────────────────────────────────

UF_REGIAO = {
    'AC': 'Norte',   'AP': 'Norte',   'AM': 'Norte',  'PA': 'Norte',
    'RO': 'Norte',   'RR': 'Norte',   'TO': 'Norte',
    'AL': 'Nordeste','BA': 'Nordeste','CE': 'Nordeste','MA': 'Nordeste',
    'PB': 'Nordeste','PE': 'Nordeste','PI': 'Nordeste','RN': 'Nordeste','SE': 'Nordeste',
    'DF': 'Centro-Oeste','GO': 'Centro-Oeste','MT': 'Centro-Oeste','MS': 'Centro-Oeste',
    'ES': 'Sudeste', 'MG': 'Sudeste', 'RJ': 'Sudeste','SP': 'Sudeste',
    'PR': 'Sul',     'RS': 'Sul',     'SC': 'Sul',
}

# ─── Keywords de busca (usadas no scraping e filtro) ─────────────────────────

KEYWORDS = [
    'violência contra a mulher', 'violência contra as mulheres',
    'violência doméstica', 'lei maria da penha', 'maria da penha',
    'feminicídio', 'femicídio', 'igualdade de gênero',
    'equidade de gênero', 'políticas para mulheres',
    'direitos das mulheres', 'direitos da mulher', 'machismo', 'misoginia',
    'sexismo', 'violência sexual', 'estupro', 'assédio sexual',
    'empoderamento feminino', 'autonomia feminina', 'discriminação de gênero',
]

# ─── Stopwords de domínio ────────────────────────────────────────────────────
# Removidas do CountVectorizer do BERTopic para que as labels dos tópicos
# mostrem o que DIFERENCIA cada tópico, não o vocabulário compartilhado.

_STOPWORDS_PARLAMENTAR = {
    # Cargos e formas de tratamento (com e sem acento, singular e plural)
    'presidente', 'presidenta',
    'deputado', 'deputada', 'deputados', 'deputadas',
    'senhor', 'senhora', 'senhores', 'senhoras',
    'senador', 'senadora', 'senadores', 'senadoras',
    'ministro', 'ministra', 'ministros', 'ministras',
    'orador', 'oradora', 'oradores',
    'vereador', 'vereadora', 'vereadores', 'vereadoras',
    'parlamentar', 'parlamentares',
    'relator', 'relatora', 'relatores',
    'sr', 'sra', 'srs', 'sras', 'excelencia',

    # Instituições
    'câmara', 'camara', 'plenário', 'plenario',
    'sessão', 'sessao', 'congresso', 'senado',
    'comissão', 'comissao', 'comissoes',
    'governo', 'secretaria', 'ministério', 'ministerio',
    'tribunal', 'justiça', 'justica',

    # Contexto geral parlamentar
    'brasil', 'brasileiro', 'brasileira', 'brasileiros', 'brasileiras',
    'estado', 'estados', 'país', 'pais', 'nação', 'nacao',
    'república', 'republica', 'federal', 'nacional',
    'constituição', 'constituicao', 'constitucional',
    'legislação', 'legislacao', 'legislatura',
    'projeto', 'projetos', 'lei', 'leis',
    'artigo', 'parágrafo', 'paragrafo', 'inciso',
    'emenda', 'votação', 'votacao', 'voto', 'votos',
    'matéria', 'materia', 'proposta', 'propostas',

    # Discurso parlamentar genérico
    'discurso', 'falar', 'dizer', 'disse', 'digo', 'diz',
    'quero', 'fazer', 'obrigado', 'obrigada',
    'bloco', 'líder', 'lider', 'partido', 'partidos',
    'revisão', 'revisao', 'bancada',
    'pública', 'publica', 'públicas', 'publicas',
    'público', 'publico', 'públicos', 'publicos',
    'política', 'politica', 'políticas', 'politicas',
    'político', 'politico', 'políticos', 'politicos',

    # Palavras genéricas frequentes em discurso
    'contra', 'não', 'nao', 'neste', 'nesta', 'nesse', 'nessa',
    'desta', 'deste', 'dessa', 'desse', 'aqui', 'hoje',
    'então', 'entao', 'porque', 'portanto',
    'também', 'tambem', 'ainda', 'apenas',
    'muito', 'muita', 'muitos', 'muitas',
    'todos', 'todas', 'todo', 'toda',
    'grande', 'grandes', 'maior', 'melhor',
    'nosso', 'nossa', 'nossos', 'nossas',
    'pode', 'podem', 'deve', 'devem',
    'seria', 'sido', 'sendo', 'haver',
    'importante', 'necessário', 'necessario',
    'momento', 'forma', 'maneira', 'modo',
    'anos', 'dias', 'tempo', 'vezes',
    'parte', 'pessoas', 'sociedade', 'população', 'populacao',
    'país', 'mundo', 'casa', 'mesa', 'povo',
    'amigos', 'amigas', 'colegas',
    'respeito', 'direito', 'direitos',
    'municipio', 'município',

    # Verbos/palavras residuais que aparecem nas labels sem agregar
    'estao', 'sao', 'vem', 'vao', 'tem', 'vamos',
    'ano', 'anos', 'dia', 'dias',
    'sobre', 'como', 'mais', 'outro', 'outra', 'outros', 'outras',
    'isso', 'isto', 'essa', 'esse', 'este', 'esta',
    'possamos', 'precisamos', 'queremos', 'temos', 'somos',
    'dizer', 'falar', 'agradecer', 'gostaria',
    'colocar', 'apresentar', 'tratar', 'trazer',
    'situacao', 'situação', 'questao', 'questão',
    'numero', 'número', 'dados', 'caso', 'casos',
    'relacao', 'relação', 'problema', 'problemas',
    'mandato', 'pauta', 'aprovacao', 'aprovação',
    'parlamento', 'plenaria', 'plenário',
}

# Keywords de busca também são stopwords do BERTopic — aparecem em 100% do
# corpus, então não diferenciam nenhum tópico.
_STOPWORDS_KEYWORDS = set()
for kw in KEYWORDS:
    _STOPWORDS_KEYWORDS.update(kw.split())

STOPWORDS_DOMAIN = (
    set(stopwords.words('portuguese'))
    | _STOPWORDS_PARLAMENTAR
    | _STOPWORDS_KEYWORDS
)
