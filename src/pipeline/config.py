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
    'REPUBLICANOS': 'Direita', 'UNIÃO': 'Direita',
    'DEM': 'Direita', 'PSC': 'Direita', 'PRB': 'Direita',
    'PTB': 'Direita', 'PR': 'Direita', 'PATRIOTA': 'Direita',
    'PFL': 'Direita',

    # Extrema-direita
    'PSL': 'Extrema-direita', 'NOVO': 'Extrema-direita',

    # Partidos extintos/históricos (presentes no CSV completo 2000-2025)
    'PTdoB': 'Centro',
    'PRONA': 'Extrema-direita',
    'PTC': 'Centro-direita', 'PRP': 'Direita',
    'PMB': 'Centro-direita',
    'PRTB': 'Extrema-direita',
    'PAN': 'Centro-esquerda',
    'PST': 'Centro-direita',
    'PHS': 'Centro-direita',
    'PPB': 'Direita', 'PPR': 'Direita',
    'PDS': 'Direita', 'ARENA': 'Direita',
    'PPL': 'Esquerda',
    'PCO': 'Extrema-esquerda', 'PSTU': 'Extrema-esquerda',

    # Fora da classificação de Bolognesi et al.: mesma classe do partido em que se transformaram
    'PTN': 'Centro-direita',        # virou PODE (2017)
    'PEN': 'Direita',               # virou PATRIOTA (2018)
    'MOBILIZA': 'Esquerda',         # ex-PMN (2023)
    'AGIR': 'Centro-direita',       # ex-PTC (2022)
    # Classificação própria, a revisar
    'PSDC': 'Centro-direita', 'DC': 'Centro-direita',  # Democracia Cristã
    'MISSÃO': 'Direita',            # Partido Missão (2025)
}

# Grafias diferentes da mesma sigla no SitaQWeb → forma usada no ESPECTRO.
# A comparação ignora maiúsculas ('PCDOB' → 'PCdoB'). Mudanças de nome (PMDB → MDB,
# PFL → DEM) não entram: são partidos diferentes no tempo.
PARTIDO_GRAFIAS = {
    'UNIAO': 'UNIÃO', 'PATRI': 'PATRIOTA', 'SDD': 'SOLIDARIEDADE', 'SOLIDARIEDAD': 'SOLIDARIEDADE',
    'PODEMOS': 'PODE', 'MISSAO': 'MISSÃO', 'S.PART.': 'SEM PARTIDO', 'S/PARTIDO': 'SEM PARTIDO',
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

    # Apareceram nos rótulos sem dizer nada sobre o tema (testes de nov/2026).
    # Verbos e advérbios já saem pelo filtro de classe gramatical (vocabulary.py);
    # ficam aqui também para não depender só do etiquetador.
    'fala', 'falas', 'falando', 'aprovar', 'aprovamos', 'infelizmente', 'existe', 'precisa',
    'votar', 'autoria', 'campanha', 'bom', 'boa', 'lá', 'la', 'junto', 'parabéns', 'parabens',
    'proposição', 'proposicao', 'proposições', 'proposicoes', 'discursos', 'relatório', 'relatorio',
    'deverá', 'devera', 'agenda', 'honra', 'fato', 'fatos', 'mesmo', 'mesma', 'mesmos', 'mesmas',
    'gente', 'presentes', 'presença', 'presenca', 'presidência', 'presidencia',
    'cumprimento', 'cumprimentos', 'discussão', 'discussao', 'debate', 'preocupação', 'preocupacao',
    # "a favor", "em favor", "por favor" (votações); "convido" e "pronunciamento" são da condução da sessão
    'favor', 'convido', 'convida', 'pronunciamento', 'pronunciamentos',
    # particípios genéricos (os particípios temáticos, como "assassinada", ficam)
    'feito', 'feita', 'feitos', 'feitas', 'sido', 'dito', 'dita', 'tido', 'dado', 'dada',
    'visto', 'vista', 'chamado', 'chamada', 'colocado', 'colocada', 'lido', 'considerado', 'considerada',
    'aprovado', 'aprovada', 'aprovados', 'aprovadas', 'divulgado', 'divulgados', 'aproveito',
}

# As keywords da busca não entram aqui: o que sai do vocabulário é o que aparece em quase todos os
# discursos (vocabulary.ubiquitous_words), o que cobre a keyword de uma busca com um termo só sem
# esconder as keywords de uma busca com vários termos.
STOPWORDS_DOMAIN = (
    set(stopwords.words('portuguese'))
    | _STOPWORDS_PARLAMENTAR
)
