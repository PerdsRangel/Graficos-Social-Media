import base64
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Impacto das Redes Sociais",
    page_icon="📱",
    layout="wide"
)

st.title("📱 O Impacto das Redes Sociais na Vida")
st.markdown(
    "Dashboard interativo para analisar os hábitos de uso das redes sociais "
    "e suas relações com sono, saúde mental, estresse e desempenho acadêmico."
)

# =====================================================================
# CONFIGURAÇÕES DE DADOS
# =====================================================================

ARQUIVO_CSV = Path(__file__).parent / "Social_media_impact_on_life.csv"

COLUNAS_NUMERICAS = [
    "Age",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
    "Sleep_Duration_Hours",
    "Sleep_Quality_Score",
    "Perceived_Stress_Score",
    "Mental_Health_Index",
    "Academic_Performance_GPA",
    "Impact_Score"   # percepção do impacto em número (0, 1, 2), criada abaixo
]

COLUNAS_CATEGORICAS = [
    "Gender",
    "Academic_Level",
    "Primary_Platform",
    "Device_Type",
    "Social_Comparison_Frequency",
    "Overall_Impact"
]

# Intervalos fisicamente/logicamente possíveis. Valores fora deles
# são erros de digitação (não outliers) e viram "ausentes".
LIMITES_VALIDOS = {
    "Age": (10, 100),
    "Daily_Usage_Hours": (0, 24),
    "Weekend_Extra_Hours": (0, 24),
    "Sleep_Duration_Hours": (0, 24),
    "Sleep_Quality_Score": (1, 5),
    "Perceived_Stress_Score": (0, 40),
    "Mental_Health_Index": (0, 100),
    "Academic_Performance_GPA": (0, 4)
}

NOMES_AMIGAVEIS = {
    "Age": "idade",
    "Weekend_Extra_Hours": "horas no fim de semana",
    "Impact_Score": "percepção do impacto",
    "Daily_Usage_Hours": "uso diário",
    "Sleep_Duration_Hours": "duração do sono",
    "Sleep_Quality_Score": "qualidade do sono",
    "Perceived_Stress_Score": "estresse",
    "Mental_Health_Index": "saúde mental",
    "Academic_Performance_GPA": "GPA"
}

VALORES_AUSENTES = ["", " ", "NA", "N/A", "n/a", "NaN", "null", "NULL", "?"]

# Valores do CSV (inglês) -> rótulos exibidos no dashboard (português)
TRADUCOES = {
    "Overall_Impact": {
        "Beneficial": "Benéfico",
        "Neutral": "Neutro",
        "Negative": "Negativo"
    },
    "Device_Type": {
        "Smartphone": "Celular",
        "Tablet": "Tablet",
        "Laptop/PC": "Computador"
    },
    "Gender": {
        "Female": "Feminino",
        "Male": "Masculino",
        "Non-Binary": "Não binário",
        "Prefer not to say": "Prefere não informar"
    },
    "Academic_Level": {
        "High School": "Ensino médio",
        "Undergraduate": "Graduação",
        "Postgraduate": "Pós-graduação"
    }
}

ORDEM_IMPACTO = ["Benéfico", "Neutro", "Negativo"]
ORDEM_GENERO = [
    "Feminino", "Masculino", "Não binário",
    "Prefere não informar", "Não informado"
]

# Grupos com menos pessoas que isso não entram nos gráficos de comparação
# (porcentagens de grupos minúsculos oscilam demais e enganam)
MIN_N = 30


# =====================================================================
# CARGA E LIMPEZA GERAL (duplicados, ausentes e valores inválidos)
# =====================================================================

@st.cache_data
def carregar_dados():
    df = pd.read_csv(ARQUIVO_CSV, na_values=VALORES_AUSENTES)
    df.columns = df.columns.str.strip()

    relatorio = {"linhas_originais": len(df)}

    # Texto: remove espaços extras e transforma vazio em ausente
    for coluna in COLUNAS_CATEGORICAS + ["Student_ID"]:
        df[coluna] = df[coluna].astype("string").str.strip()
        df[coluna] = df[coluna].replace("", pd.NA)

    # Traduz para português os campos que aparecem nos gráficos
    for coluna, mapa in TRADUCOES.items():
        df[coluna] = df[coluna].replace(mapa)

    # Percepção do impacto em número (necessário para as correlações)
    df["Impact_Score"] = df["Overall_Impact"].map(
        {"Negativo": 0, "Neutro": 1, "Benéfico": 2}
    )

    # Números: converte e invalida o que é impossível
    for coluna in COLUNAS_NUMERICAS:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce").astype(float)

    invalidos = 0
    for coluna, (minimo, maximo) in LIMITES_VALIDOS.items():
        fora = df[coluna].notna() & ~df[coluna].between(minimo, maximo)
        invalidos += int(fora.sum())
        df.loc[fora, coluna] = float("nan")
    relatorio["valores_invalidos"] = invalidos

    # Duplicados: linha idêntica OU mesmo Student_ID (mantém o primeiro)
    antes = len(df)
    df = df.drop_duplicates()
    df = df[df["Student_ID"].isna() | ~df["Student_ID"].duplicated()]
    relatorio["duplicados_removidos"] = antes - len(df)

    # Categorias ausentes viram "Não informado" (não perde o participante)
    relatorio["categorias_preenchidas"] = int(
        df[COLUNAS_CATEGORICAS].isna().sum().sum()
    )
    for coluna in COLUNAS_CATEGORICAS:
        df[coluna] = df[coluna].astype(object).fillna("Não informado")

    df = df.reset_index(drop=True)
    relatorio["linhas_finais"] = len(df)
    relatorio["ausentes_por_coluna"] = {
        c: int(n) for c, n in df[COLUNAS_NUMERICAS].isna().sum().items() if n > 0
    }

    return df, relatorio


df, relatorio_carga = carregar_dados()

# Mediana por nível acadêmico (usada apenas se o usuário escolher imputar)
MEDIANAS_NIVEL = df.groupby("Academic_Level")[COLUNAS_NUMERICAS].median()
MEDIANAS_GLOBAIS = df[COLUNAS_NUMERICAS].median()


# =====================================================================
# MODOS PARA DALTONISMO
# =====================================================================
# Protanomalia/Protanopia/Deuteranomalia/Deuteranopia: confusão
#   vermelho-verde -> paleta Okabe-Ito (azul/laranja/amarelo...).
# Tritanomalia/Tritanopia: confusão azul-amarelo -> vermelho/azul-petróleo/roxo.
# Monocromacia: só luminosidade -> tons de cinza + padrões e símbolos.
# As "anomalias" (visão parcial) usam as mesmas paletas das "anopias"
# (ausência total do cone), que também funcionam para o caso parcial.

PALETA_OKABE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#F0E442", "#D55E00"
]
PALETA_TRITAN = [
    "#C1272D", "#00A19B", "#7B3294", "#222222",
    "#F4A6C6", "#8C8C8C", "#FFD92F"
]
PALETA_CINZA = [
    "#000000", "#9A9A9A", "#4D4D4D", "#C8C8C8",
    "#2B2B2B", "#7A7A7A", "#B0B0B0"
]

_OKABE = dict(
    discreta=PALETA_OKABE,
    escala="Cividis",
    impacto={
        "Benéfico": "#0072B2",
        "Neutro": "#999999",
        "Negativo": "#E69F00"
    },
    tendencia="#000000",
    contorno="rgba(0,0,0,0.55)",
    divergente=["#E69F00", "#FFFFFF", "#0072B2"],
)
_TRITAN = dict(
    discreta=PALETA_TRITAN,
    escala=["#3B0F70", "#8C2981", "#DE4968", "#FE9F6D"],
    impacto={
        "Benéfico": "#00A19B",
        "Neutro": "#8C8C8C",
        "Negativo": "#C1272D"
    },
    tendencia="#000000",
    contorno="rgba(0,0,0,0.55)",
    divergente=["#C1272D", "#FFFFFF", "#00A19B"],
)
_MONO = dict(
    discreta=PALETA_CINZA,
    escala=["#E6E6E6", "#8C8C8C", "#000000"],
    impacto={
        "Benéfico": "#000000",
        "Neutro": "#9A9A9A",
        "Negativo": "#E0E0E0"
    },
    tendencia="#000000",
    contorno="#000000",
    divergente=["#3A3A3A", "#FFFFFF", "#3A3A3A"],
)

MODOS_VISAO = {
    "Visão normal": dict(
        desc="Cores padrão do Plotly.",
        discreta=px.colors.qualitative.Plotly,
        escala="Turbo",
        impacto={
            "Benéfico": "#00CC96",
            "Neutro": "#636EFA",
            "Negativo": "#EF553B"
        },
        tendencia="#EF553B",
        contorno="white",
        divergente="RdBu",
    ),
    "Protanomalia": dict(
        desc="Vermelho enfraquecido (parcial). Paleta azul/laranja/verde-azulado, "
             "sem depender de vermelho vs. verde.", **_OKABE),
    "Protanopia": dict(
        desc="Ausência de cone vermelho. Vermelho parece escuro; usa-se "
             "azul/laranja/amarelo.", **_OKABE),
    "Deuteranomalia": dict(
        desc="Verde enfraquecido (parcial, o tipo mais comum). Paleta "
             "azul/laranja.", **_OKABE),
    "Deuteranopia": dict(
        desc="Ausência de cone verde. Paleta azul/laranja/amarelo.", **_OKABE),
    "Tritanomalia": dict(
        desc="Azul enfraquecido (parcial). Evita azul vs. verde e amarelo vs. "
             "rosa; usa vermelho/azul-petróleo/roxo.", **_TRITAN),
    "Tritanopia": dict(
        desc="Ausência de cone azul. Paleta vermelho/azul-petróleo/roxo.",
        **_TRITAN),
    "Monocromacia": dict(
        desc="Sem percepção de cor (só claro/escuro). Tons de cinza com "
             "padrões e símbolos obrigatórios.", **_MONO),
}

PADROES = ["", "/", "\\", "x", "-", "|", "+"]
MAPA_PADRAO_IMPACTO = {
    "Benéfico": "/",
    "Neutro": "x",
    "Negativo": "-"
}
SIMBOLOS_IMPACTO = {
    "Benéfico": "circle",
    "Neutro": "square",
    "Negativo": "diamond"
}

# =====================================================================
# BARRA LATERAL: filtros, tratamento e acessibilidade
# =====================================================================

st.sidebar.header("⚙️ Filtros")

idade_min = int(df["Age"].min())
idade_max = int(df["Age"].max())

idade = st.sidebar.slider(
    "Faixa etária",
    idade_min,
    idade_max,
    (idade_min, idade_max)
)

generos = sorted(df["Gender"].dropna().unique())
genero = st.sidebar.multiselect("Gênero", generos, default=generos)

plataformas = sorted(df["Primary_Platform"].dropna().unique())
plataforma = st.sidebar.multiselect(
    "Plataforma principal", plataformas, default=plataformas
)

niveis = sorted(df["Academic_Level"].dropna().unique())
nivel = st.sidebar.multiselect("Nível acadêmico", niveis, default=niveis)

impactos = sorted(df["Overall_Impact"].dropna().unique())
impacto = st.sidebar.multiselect(
    "Impacto percebido", impactos, default=impactos
)

st.sidebar.divider()
st.sidebar.header("🧹 Tratamento de dados")

MODO_AUSENTES = st.sidebar.radio(
    "Valores ausentes",
    ["Remover linhas incompletas", "Imputar pela mediana do nível acadêmico"],
    help="Aplicado em cada gráfico apenas às colunas que ele usa."
)
MODO_OUTLIERS = st.sidebar.radio(
    "Outliers (método IQR)",
    ["Remover", "Limitar aos limites (winsorização)", "Manter"],
    help="Outlier = valor fora de [Q1 − k·IQR, Q3 + k·IQR], calculado sobre "
         "os dados filtrados e apenas nas colunas usadas em cada gráfico."
)
FATOR_IQR = st.sidebar.slider(
    "Fator k do IQR", 1.0, 3.0, 1.5, 0.5,
    help="1,5 é o padrão; 3,0 considera apenas outliers extremos."
)

with st.sidebar.expander("📋 Qualidade dos dados (carga)"):
    st.write(f"Linhas no arquivo: **{relatorio_carga['linhas_originais']:,}**")
    st.write(
        f"Duplicados removidos: **{relatorio_carga['duplicados_removidos']}**"
    )
    st.write(
        f"Valores impossíveis tornados ausentes: "
        f"**{relatorio_carga['valores_invalidos']}**"
    )
    st.write(
        f"Categorias vazias → 'Não informado': "
        f"**{relatorio_carga['categorias_preenchidas']}**"
    )
    st.write(f"Linhas finais: **{relatorio_carga['linhas_finais']:,}**")
    if relatorio_carga["ausentes_por_coluna"]:
        st.write("Ausentes numéricos por coluna:")
        st.dataframe(
            pd.Series(relatorio_carga["ausentes_por_coluna"], name="Ausentes"),
            use_container_width=True
        )

st.sidebar.divider()
st.sidebar.header("👁️ Modo daltônico")

nome_modo = st.sidebar.selectbox("Tipo de visão", list(MODOS_VISAO.keys()))
MODO = MODOS_VISAO[nome_modo]
st.sidebar.caption(MODO["desc"])

PADROES_ATIVOS = st.sidebar.checkbox(
    "Reforçar com padrões e símbolos",
    value=False,
    help="Adiciona texturas nas barras e formatos nos pontos, para que a "
         "informação não dependa só da cor."
) or nome_modo == "Monocromacia"
if nome_modo == "Monocromacia":
    st.sidebar.caption("Padrões e símbolos ativados automaticamente.")

amostras = "".join(
    f"<span style='display:inline-block;width:22px;height:22px;"
    f"background:{cor};border:1px solid #888;margin-right:3px;"
    f"border-radius:4px'></span>"
    for cor in MODO["discreta"]
)
st.sidebar.markdown(amostras, unsafe_allow_html=True)

# Mapas de cor/padrão fixos (a mesma plataforma tem sempre a mesma cor)
PALETA = MODO["discreta"]
MAPA_COR_PLATAFORMA = {
    p: PALETA[i % len(PALETA)] for i, p in enumerate(plataformas)
}
MAPA_PADRAO_PLATAFORMA = {
    p: PADROES[i % len(PADROES)] for i, p in enumerate(plataformas)
}
MAPA_COR_IMPACTO = MODO["impacto"]
_generos_cores = [g for g in ORDEM_GENERO if g in set(df["Gender"])]
MAPA_COR_GENERO = {
    g: PALETA[i % len(PALETA)] for i, g in enumerate(_generos_cores)
}
MAPA_PADRAO_GENERO = {
    g: PADROES[i % len(PADROES)] for i, g in enumerate(_generos_cores)
}


def ajustar_barras(fig):
    """Contorno nas barras para reforçar o limite entre categorias."""
    if PADROES_ATIVOS or nome_modo != "Visão normal":
        fig.update_traces(
            marker_line_color=MODO["contorno"] if nome_modo != "Visão normal"
            else "#333333",
            marker_line_width=1
        )
    return fig


# =====================================================================
# FILTRAGEM
# =====================================================================

filtrado = df[
    (df["Age"].between(idade[0], idade[1])) &
    (df["Gender"].isin(genero)) &
    (df["Primary_Platform"].isin(plataforma)) &
    (df["Academic_Level"].isin(nivel)) &
    (df["Overall_Impact"].isin(impacto))
].copy()

if filtrado.empty:
    st.warning("Nenhum dado encontrado com os filtros selecionados.")
    st.stop()


# =====================================================================
# FUNÇÕES DE TRATAMENTO (aplicadas gráfico a gráfico)
# =====================================================================

def tratar(dados, colunas_num=None, colunas_cat=None, outliers=True,
           sem_outliers=None):
    """
    Trata os dados de UM gráfico, usando só as colunas que ele consome:
      1) Duplicados  -> remove Student_ID repetido
      2) Ausentes    -> categóricas: 'Não informado'; numéricas: remove
                        a linha ou imputa a mediana do nível acadêmico
      3) Outliers    -> IQR: remove, limita (winsoriza) ou mantém
                        (outliers=False desliga esta etapa, ex.: idade,
                        em que valores altos são reais)
    Retorna (dataframe tratado, relatório).
    """
    colunas_num = colunas_num or []
    colunas_cat = colunas_cat or []
    sem_outliers = sem_outliers or []

    d = dados.copy()
    rel = {"colunas": colunas_num, "n_inicial": len(d)}

    # 1) duplicados
    antes = len(d)
    d = d.drop_duplicates(subset=["Student_ID"])
    rel["duplicados"] = antes - len(d)

    # 2) ausentes
    rel["cat_preenchidas"] = 0
    for c in colunas_cat:
        rel["cat_preenchidas"] += int(d[c].isna().sum())
        d[c] = d[c].fillna("Não informado")

    rel["ausentes"] = 0
    rel["modo_ausentes"] = MODO_AUSENTES
    if colunas_num:
        incompleta = d[colunas_num].isna().any(axis=1)
        rel["ausentes"] = int(incompleta.sum())
        if MODO_AUSENTES.startswith("Remover"):
            d = d[~incompleta]
        else:
            for c in colunas_num:
                d[c] = d[c].fillna(d["Academic_Level"].map(MEDIANAS_NIVEL[c]))
                d[c] = d[c].fillna(MEDIANAS_GLOBAIS[c])

    # 3) outliers (IQR)
    rel["outliers"] = 0
    rel["modo_outliers"] = MODO_OUTLIERS
    rel["outliers_ativo"] = outliers
    rel["sem_outliers"] = [c for c in sem_outliers if c in colunas_num]
    if outliers and colunas_num and len(d) >= 4:
        mascara = pd.Series(False, index=d.index)
        for c in colunas_num:
            if c in sem_outliers:
                continue
            q1, q3 = d[c].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            inferior = q1 - FATOR_IQR * iqr
            superior = q3 + FATOR_IQR * iqr
            mascara |= (d[c] < inferior) | (d[c] > superior)
            if MODO_OUTLIERS.startswith("Limitar"):
                d[c] = d[c].clip(inferior, superior)
        rel["outliers"] = int(mascara.sum())
        if MODO_OUTLIERS == "Remover":
            d = d[~mascara]

    rel["n_final"] = len(d)
    return d, rel


def descrever(rel):
    nome = ", ".join(NOMES_AMIGAVEIS.get(c, c) for c in rel["colunas"])
    partes = [f"duplicados removidos: {rel['duplicados']}"]

    if rel["colunas"]:
        if rel["modo_ausentes"].startswith("Remover"):
            partes.append(f"linhas com ausentes removidas: {rel['ausentes']}")
        else:
            partes.append(f"linhas com ausentes imputadas: {rel['ausentes']}")

        if rel.get("outliers_ativo", True):
            verbo = {
                "Remover": "removidos",
                "Manter": "detectados e mantidos"
            }.get(rel["modo_outliers"], "limitados")
            partes.append(f"outliers {verbo}: {rel['outliers']}")
            if rel.get("sem_outliers"):
                ignoradas = ", ".join(
                    NOMES_AMIGAVEIS.get(c, c) for c in rel["sem_outliers"]
                )
                partes.append(f"outliers não avaliados em: {ignoradas}")
        else:
            partes.append(
                "outliers: não aplicado (valores altos são reais)"
            )
    else:
        partes.append(
            f"categorias vazias preenchidas: {rel['cat_preenchidas']}"
        )
        partes.append("outliers: não se aplica (variável categórica)")

    prefixo = f"[{nome}] " if nome else ""
    return (
        f"🧹 {prefixo}" + " · ".join(partes) +
        f" · registros usados: {rel['n_final']:,} de {rel['n_inicial']:,}"
    )


def mostrar_relatorio(*rels):
    st.caption("  \n".join(descrever(r) for r in rels))


def sem_dados(dados, minimo=1):
    if len(dados) < minimo:
        st.warning(
            "Dados insuficientes para este gráfico depois do tratamento. "
            "Tente relaxar os filtros ou o tratamento de outliers."
        )
        return True
    return False


# =====================================================================
# FORMATAÇÃO EM PORTUGUÊS
# =====================================================================

def inteiro_br(n):
    return f"{int(n):,}".replace(",", ".")


def numero_br(v, casas=1, unidade=""):
    return f"{v:.{casas}f}".replace(".", ",") + unidade


def porcentagem_br(v):
    return (f"{v:.0f}%" if v >= 10 else f"{v:.1f}%").replace(".", ",")


def sinal_br(v, casas=2):
    """Número com sinal explícito (+0,62 / −0,35) para correlações."""
    sinal = "+" if v > 0 else ("−" if v < 0 else "")
    return sinal + f"{abs(v):.{casas}f}".replace(".", ",")


# =====================================================================
# CORRELAÇÃO
# =====================================================================

def forca_correlacao(v):
    """Classifica o tamanho da relação (vale para r, rho e V de Cramér)."""
    a = abs(v)
    if a < 0.10:
        return "praticamente nenhuma"
    if a < 0.30:
        return "fraca"
    if a < 0.50:
        return "moderada"
    if a < 0.70:
        return "forte"
    return "muito forte"


def sentido_correlacao(v):
    return "positiva" if v > 0 else "negativa"


def cramer_v(a, b):
    """
    V de Cramér: força da associação entre duas variáveis de categorias
    (ex.: aparelho × percepção). Vai de 0 (nenhuma relação) a 1 (total).
    """
    tabela = pd.crosstab(a, b)
    if tabela.shape[0] < 2 or tabela.shape[1] < 2:
        return float("nan")
    n = tabela.values.sum()
    esperado = np.outer(tabela.sum(axis=1), tabela.sum(axis=0)) / n
    qui2 = ((tabela.values - esperado) ** 2 / esperado).sum()
    return float(np.sqrt(qui2 / (n * (min(tabela.shape) - 1))))


# =====================================================================
# IMAGENS E ÍCONES
# =====================================================================
# Prioridade: 1) imagem sua na pasta "imagens/" (png, jpg, webp, svg, gif)
#             2) ícone desenhado em SVG, pintado com a cor do modo
#                daltônico escolhido.
#
# Nomes de arquivo aceitos em "imagens/":
#   celular, tablet, computador, feminino, masculino, nao_binario,
#   prefere_nao_informar, nao_informado, beneficio, neutro, negativo

PASTA_IMAGENS = Path(__file__).parent / "imagens"
EXTENSOES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".gif": "image/gif", ".svg": "image/svg+xml"
}
ARQUIVOS_IMAGEM = {
    "Celular": "celular",
    "Tablet": "tablet",
    "Computador": "computador",
    "Feminino": "feminino",
    "Masculino": "masculino",
    "Não binário": "nao_binario",
    "Prefere não informar": "prefere_nao_informar",
    "Não informado": "nao_informado",
    "Benéfico": "beneficio",
    "Neutro": "neutro",
    "Negativo": "negativo"
}

_TRACO = 'fill="none" stroke="{c}" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"'
_ROSTO = (
    '<circle cx="32" cy="32" r="27" ' + _TRACO + '/>'
    '<circle cx="23" cy="26" r="3.5" fill="{c}"/>'
    '<circle cx="41" cy="26" r="3.5" fill="{c}"/>'
)

ICONES_SVG = {
    "Celular": (
        '<rect x="18" y="4" width="28" height="56" rx="6" fill="{c}"/>'
        '<rect x="22" y="11" width="20" height="38" rx="2" fill="#fff"/>'
        '<circle cx="32" cy="54.5" r="2.2" fill="#fff"/>'
    ),
    "Tablet": (
        '<rect x="6" y="10" width="52" height="44" rx="6" fill="{c}"/>'
        '<rect x="11" y="15" width="42" height="34" rx="2" fill="#fff"/>'
    ),
    "Computador": (
        '<rect x="10" y="10" width="44" height="32" rx="3" fill="{c}"/>'
        '<rect x="14" y="14" width="36" height="24" rx="1" fill="#fff"/>'
        '<path d="M3 46h58l-4 8H7z" fill="{c}"/>'
    ),
    "Feminino": (
        '<circle cx="32" cy="22" r="14" ' + _TRACO + '/>'
        '<path d="M32 36v22M22 48h20" ' + _TRACO + '/>'
    ),
    "Masculino": (
        '<circle cx="26" cy="38" r="14" ' + _TRACO + '/>'
        '<path d="M36 28L54 10M42 10h12v12" ' + _TRACO + '/>'
    ),
    "Não binário": (
        '<circle cx="30" cy="32" r="12" ' + _TRACO + '/>'
        '<path d="M30 44v16M22 53h16M39 23L53 9M44 9h9v9" ' + _TRACO + '/>'
    ),
    "Prefere não informar": (
        '<circle cx="32" cy="32" r="27" ' + _TRACO + '/>'
        '<path d="M24 25c0-5 4-8 8-8s8 3 8 7c0 5-8 6-8 12" ' + _TRACO + '/>'
        '<circle cx="32" cy="46" r="2.8" fill="{c}"/>'
    ),
    "Benéfico": _ROSTO + '<path d="M19 39Q32 53 45 39" ' + _TRACO + '/>',
    "Neutro": _ROSTO + '<path d="M21 43H43" ' + _TRACO + '/>',
    "Negativo": _ROSTO + '<path d="M19 48Q32 35 45 48" ' + _TRACO + '/>',
}
ICONES_SVG["Não informado"] = ICONES_SVG["Prefere não informar"]


def _cor_com_contraste(cor):
    """Escurece cores muito claras para o ícone não sumir no cartão claro."""
    if not (isinstance(cor, str) and cor.startswith("#") and len(cor) == 7):
        return cor
    r, g, b = (int(cor[i:i + 2], 16) / 255 for i in (1, 3, 5))
    luminancia = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#555555" if luminancia > 0.7 else cor


@lru_cache(maxsize=256)
def imagem_uri(rotulo, cor):
    """Devolve a imagem do rótulo como data URI (sua imagem ou o SVG)."""
    base = ARQUIVOS_IMAGEM.get(rotulo)
    if base:
        for extensao, mime in EXTENSOES.items():
            caminho = PASTA_IMAGENS / f"{base}{extensao}"
            if caminho.exists():
                dados = base64.b64encode(caminho.read_bytes()).decode()
                return f"data:{mime};base64,{dados}"

    corpo = ICONES_SVG.get(rotulo)
    if corpo is None:
        return None
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        + corpo.replace("{c}", _cor_com_contraste(cor)) + "</svg>"
    )
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def mostrar_cartoes(itens):
    """
    Mostra uma fileira de cartões com imagem.
    itens: lista de (rótulo, linha 1, linha 2, cor do ícone).
    """
    colunas_cartoes = st.columns(len(itens))
    for coluna_st, (rotulo, linha1, linha2, cor) in zip(colunas_cartoes, itens):
        uri = imagem_uri(rotulo, cor)
        imagem = (
            f'<img src="{uri}" alt="{rotulo}" '
            f'style="height:64px;max-width:100%;object-fit:contain"/>'
            if uri else ""
        )
        coluna_st.markdown(
            '<div style="text-align:center;padding:14px 8px;'
            'border-radius:14px;background:#F5F7FA;color:#1F2937;'
            'border:1px solid #D5DAE1;">'
            f'{imagem}'
            f'<div style="font-weight:700;font-size:1.05rem;margin-top:6px">'
            f'{rotulo}</div>'
            f'<div style="font-size:.92rem">{linha1}</div>'
            f'<div style="font-size:.8rem;color:#4B5563">{linha2}</div>'
            '</div>',
            unsafe_allow_html=True
        )


# =====================================================================
# INDICADORES
# =====================================================================

st.subheader("📊 Indicadores gerais")


def media_tratada(coluna):
    d, rel = tratar(filtrado, [coluna])
    valor = d[coluna].mean() if len(d) else float("nan")
    return valor, rel


def formatar(valor, sufixo=""):
    return "—" if pd.isna(valor) else f"{valor:.1f}{sufixo}"


uso, rel_uso = media_tratada("Daily_Usage_Hours")
sono, rel_sono = media_tratada("Sleep_Duration_Hours")
estresse, rel_estresse = media_tratada("Perceived_Stress_Score")
saude, rel_saude = media_tratada("Mental_Health_Index")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Participantes", f"{len(filtrado):,}")
c2.metric("Uso diário médio", formatar(uso, " h"))
c3.metric("Sono médio", formatar(sono, " h"))
c4.metric("Estresse médio", formatar(estresse))
c5.metric("Saúde mental", formatar(saude))

st.caption(
    "Os indicadores apresentam as médias dos participantes "
    "considerados após a aplicação dos filtros. Cada média é calculada "
    "de forma independente, com o tratamento de ausentes e outliers "
    "escolhido na barra lateral."
)
mostrar_relatorio(rel_uso, rel_sono, rel_estresse, rel_saude)

st.divider()

# =====================================================================
# 1º GRÁFICO — Plataformas (categórico)
# =====================================================================

st.subheader("📱 Plataformas mais utilizadas")

d1, rel1 = tratar(filtrado, colunas_cat=["Primary_Platform"])

if not sem_dados(d1):
    dados = (
        d1["Primary_Platform"]
        .value_counts()
        .rename_axis("Plataforma")
        .reset_index(name="Participantes")
    )

    fig = px.bar(
        dados,
        x="Plataforma",
        y="Participantes",
        color="Plataforma",
        color_discrete_map=MAPA_COR_PLATAFORMA,
        pattern_shape="Plataforma" if PADROES_ATIVOS else None,
        pattern_shape_map=MAPA_PADRAO_PLATAFORMA,
        text="Participantes",
        title="Distribuição das plataformas utilizadas"
    )
    fig.update_traces(textposition="outside")
    ajustar_barras(fig)

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Este gráfico mostra a quantidade de participantes que "
        "utilizam cada plataforma como sua principal rede social. "
        "As maiores barras representam as plataformas mais populares "
        "entre os participantes."
    )
    mostrar_relatorio(rel1)

# =====================================================================
# 2º GRÁFICO — Percepção do impacto (categórico)
# =====================================================================

st.subheader("🌐 Percepção do impacto das redes sociais")

d2, rel2 = tratar(filtrado, colunas_cat=["Overall_Impact"])

if not sem_dados(d2):
    dados = (
        d2["Overall_Impact"]
        .value_counts()
        .rename_axis("Impacto")
        .reset_index(name="Participantes")
    )

    total2 = dados["Participantes"].sum()
    contagem2 = dados.set_index("Impacto")["Participantes"]
    mostrar_cartoes([
        (
            i,
            f"{inteiro_br(contagem2[i])} pessoas",
            f"{porcentagem_br(contagem2[i] / total2 * 100)} do total",
            MAPA_COR_IMPACTO.get(i, "#777777")
        )
        for i in ORDEM_IMPACTO + ["Não informado"] if i in contagem2.index
    ])

    fig = px.pie(
        dados,
        names="Impacto",
        values="Participantes",
        hole=0.45,
        color="Impacto",
        color_discrete_map=MAPA_COR_IMPACTO,
        category_orders={"Impacto": ORDEM_IMPACTO},
        title="Percepção geral dos participantes"
    )
    # Rótulos escritos direto nas fatias: não dependem da cor
    fig.update_traces(
        textinfo="label+percent",
        textposition="outside",
        marker=dict(line=dict(color=MODO["contorno"], width=2))
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "O gráfico apresenta a proporção de participantes que "
        "consideram o impacto das redes sociais positivo, neutro "
        "ou negativo. Ele permite visualizar a percepção geral "
        "dos entrevistados."
    )
    mostrar_relatorio(rel2)

# =====================================================================
# 3º GRÁFICO — Histograma do uso diário
# =====================================================================

st.subheader("⏱️ Distribuição do tempo diário de uso")

d3, rel3 = tratar(filtrado, ["Daily_Usage_Hours"])

if not sem_dados(d3):
    fig = px.histogram(
        d3,
        x="Daily_Usage_Hours",
        nbins=25,
        color_discrete_sequence=[PALETA[0]],
        title="Horas diárias nas redes sociais"
    )
    fig.update_traces(
        marker_line_color="#333333" if nome_modo != "Visão normal" else "white",
        marker_line_width=0.6
    )
    fig.update_layout(
        xaxis_title="Horas de uso por dia",
        yaxis_title="Quantidade de participantes"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Este histograma apresenta a distribuição das horas "
        "diárias de utilização das redes sociais. As barras mais "
        "altas indicam os intervalos de tempo mais frequentes."
    )
    mostrar_relatorio(rel3)

# =====================================================================
# 4º GRÁFICO — Uso das redes × bem-estar
# (junta os antigos gráficos 4, 5 e 6 em um só, mais simples)
# =====================================================================

INDICADORES = {
    "Horas de sono por noite": dict(
        col="Sleep_Duration_Hours", unidade=" h", casas=1,
        curto="horas de sono", sentido="quanto maior, mais a pessoa dorme"
    ),
    "Qualidade do sono (nota de 1 a 5)": dict(
        col="Sleep_Quality_Score", unidade="", casas=1,
        curto="qualidade do sono", sentido="quanto maior, melhor o sono"
    ),
    "Nível de estresse (0 a 40)": dict(
        col="Perceived_Stress_Score", unidade="", casas=1,
        curto="nível de estresse", sentido="quanto maior, mais estresse"
    ),
    "Saúde mental (0 a 100)": dict(
        col="Mental_Health_Index", unidade="", casas=0,
        curto="índice de saúde mental", sentido="quanto maior, melhor"
    ),
    "Nota na escola/faculdade (0 a 4)": dict(
        col="Academic_Performance_GPA", unidade="", casas=2,
        curto="nota acadêmica", sentido="quanto maior, melhor a nota"
    ),
}

FAIXAS_USO = [0, 3, 5, 7, 9, float("inf")]
ROTULOS_USO = ["Até 3 h", "De 3 a 5 h", "De 5 a 7 h", "De 7 a 9 h", "Mais de 9 h"]


st.subheader("🔎 Quem usa mais as redes é diferente de quem usa menos?")
st.markdown(
    "Escolha **o que comparar** e **como separar as pessoas em grupos**. "
    "Cada barra mostra a **média** de um grupo (embaixo de cada barra "
    "está quantas pessoas formam o grupo)."
)

escolha_a, escolha_b = st.columns(2)
nome_indicador = escolha_a.selectbox(
    "O que você quer comparar?", list(INDICADORES.keys())
)
agrupar_por = escolha_b.radio(
    "Separar as pessoas por",
    ["Tempo de uso por dia", "Rede social principal"],
    horizontal=True
)

info = INDICADORES[nome_indicador]
coluna = info["col"]

d4, rel4 = tratar(
    filtrado, ["Daily_Usage_Hours", coluna], ["Primary_Platform"]
)

if not sem_dados(d4, minimo=MIN_N):

    por_horas = agrupar_por == "Tempo de uso por dia"

    if por_horas:
        d4["Grupo"] = pd.cut(
            d4["Daily_Usage_Hours"],
            bins=FAIXAS_USO,
            labels=ROTULOS_USO,
            include_lowest=True
        )
        titulo_x = "Quanto tempo a pessoa passa nas redes por dia"
    else:
        d4["Grupo"] = d4["Primary_Platform"]
        titulo_x = "Rede social que a pessoa mais usa"

    tabela = (
        d4.groupby("Grupo", observed=True)
        .agg(media=(coluna, "mean"), pessoas=(coluna, "size"))
        .reset_index()
    )
    tabela["Grupo"] = tabela["Grupo"].astype(str)

    omitidos = tabela[tabela["pessoas"] < MIN_N]["Grupo"].tolist()
    tabela = tabela[tabela["pessoas"] >= MIN_N]

    if not por_horas:
        tabela = tabela.sort_values("media", ascending=False)

    if not sem_dados(tabela, minimo=2):
        tabela["Rótulo"] = (
            tabela["Grupo"] + "<br>(" +
            tabela["pessoas"].map(inteiro_br) + " pessoas)"
        )
        tabela["Texto"] = tabela["media"].map(
            lambda v: numero_br(v, info["casas"], info["unidade"])
        )

        fig = px.bar(
            tabela,
            x="Rótulo",
            y="media",
            text="Texto",
            color_discrete_sequence=[PALETA[0]],
            category_orders={"Rótulo": tabela["Rótulo"].tolist()},
            hover_data={"pessoas": True, "Rótulo": False, "media": False},
            title=f"{nome_indicador} — média de cada grupo"
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        ajustar_barras(fig)
        fig.update_layout(
            xaxis_title=titulo_x,
            yaxis_title=f"Média ({info['sentido']})",
            yaxis_range=[0, tabela["media"].max() * 1.18],
            font_size=14,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Correlação entre horas de uso e o indicador (todas as pessoas)
        r_p = d4["Daily_Usage_Hours"].corr(d4[coluna])
        r_s = d4["Daily_Usage_Hours"].corr(d4[coluna], method="spearman")
        if pd.notna(r_p) and pd.notna(r_s):
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Correlação de Pearson",
                sinal_br(r_p),
                help="De −1 a +1. Mede o quanto as duas coisas variam "
                     "juntas em linha reta. 0 = sem relação."
            )
            m2.metric(
                "Correlação de Spearman",
                sinal_br(r_s),
                help="Parecida com a de Pearson, mas compara a ordem dos "
                     "valores; resiste melhor a valores extremos."
            )
            m3.metric(
                "Força da relação",
                forca_correlacao(r_p).capitalize(),
                help="Classificação do valor de Pearson: até 0,10 "
                     "praticamente nenhuma; até 0,30 fraca; até 0,50 "
                     "moderada; até 0,70 forte; acima disso muito forte."
            )

        # Resumo em linguagem simples, calculado a partir dos próprios dados
        primeiro = tabela.iloc[0]
        ultimo = tabela.iloc[-1]
        v1 = numero_br(primeiro["media"], info["casas"], info["unidade"])
        v2 = numero_br(ultimo["media"], info["casas"], info["unidade"])
        diferenca = abs(primeiro["media"] - ultimo["media"])
        parecidas = diferenca < 0.15 * d4[coluna].std()

        if por_horas:
            resumo = (
                f"👉 **Em resumo:** quem fica **{primeiro['Grupo'].lower()}** "
                f"por dia nas redes tem, em média, **{v1}** "
                f"({info['curto']}). Quem fica "
                f"**{ultimo['Grupo'].lower()}** tem **{v2}**."
            )
        else:
            resumo = (
                f"👉 **Em resumo:** a rede com a **maior** média de "
                f"{info['curto']} é o **{primeiro['Grupo']}** ({v1}) e "
                f"a **menor** é o **{ultimo['Grupo']}** ({v2})."
            )
        if parecidas:
            resumo += (
                " As médias são muito parecidas: esta divisão quase não "
                "muda o resultado."
            )
        if pd.notna(r_p):
            if abs(r_p) < 0.10:
                resumo += (
                    f"  \n📈 **Correlação:** praticamente nenhuma relação "
                    f"entre as horas de uso e {info['curto']} "
                    f"(r = {sinal_br(r_p)})."
                )
            else:
                tende = "maior" if r_p > 0 else "menor"
                resumo += (
                    f"  \n📈 **Correlação:** relação "
                    f"**{forca_correlacao(r_p)} e {sentido_correlacao(r_p)}** "
                    f"(r = {sinal_br(r_p)}): quanto mais horas de uso por "
                    f"dia, {info['curto']} costuma ser **{tende}**."
                )
        st.info(resumo)

        st.caption(
            f"Como ler: {info['sentido']}. Isso mostra uma **associação**, "
            "não prova que o tempo de uso causa a diferença."
        )
        if omitidos:
            st.caption(
                f"Grupos com menos de {MIN_N} pessoas foram omitidos: "
                f"{', '.join(omitidos)}."
            )
    mostrar_relatorio(rel4)

# =====================================================================
# 5º GRÁFICO — Impacto percebido por tipo de aparelho
# =====================================================================

st.subheader("📲 O aparelho usado muda a percepção do impacto?")

d5, rel5 = tratar(filtrado, colunas_cat=["Device_Type", "Overall_Impact"])

if not sem_dados(d5, minimo=MIN_N):

    contagem = pd.crosstab(d5["Device_Type"], d5["Overall_Impact"])
    totais = contagem.sum(axis=1)
    porcentagem = contagem.div(totais, axis=0) * 100

    longo = (
        porcentagem.reset_index()
        .melt(id_vars="Device_Type", var_name="Percepção",
              value_name="Porcentagem")
    )
    longo["Pessoas"] = longo["Device_Type"].map(totais)
    longo = longo[longo["Pessoas"] >= MIN_N]

    if not sem_dados(longo):
        ordem_aparelhos = (
            totais[totais >= MIN_N].sort_values(ascending=False).index.tolist()
        )
        total5 = totais[ordem_aparelhos].sum()
        mostrar_cartoes([
            (
                a,
                f"{inteiro_br(totais[a])} pessoas",
                f"{porcentagem_br(totais[a] / total5 * 100)} do total",
                PALETA[0]
            )
            for a in ordem_aparelhos
        ])
        longo["Rótulo"] = (
            longo["Device_Type"] + "<br>(" +
            longo["Pessoas"].map(inteiro_br) + " pessoas)"
        )
        ordem_rotulos = [
            f"{a}<br>({inteiro_br(totais[a])} pessoas)"
            for a in ordem_aparelhos
        ]
        longo["Texto"] = longo["Porcentagem"].map(porcentagem_br)

        fig = px.bar(
            longo,
            x="Rótulo",
            y="Porcentagem",
            color="Percepção",
            color_discrete_map=MAPA_COR_IMPACTO,
            pattern_shape="Percepção" if PADROES_ATIVOS else None,
            pattern_shape_map=MAPA_PADRAO_IMPACTO,
            category_orders={
                "Rótulo": ordem_rotulos,
                "Percepção": ORDEM_IMPACTO
            },
            text="Texto",
            barmode="group",
            hover_data={"Pessoas": True, "Rótulo": False},
            title="De cada 100 pessoas, quantas acham o impacto "
                  "benéfico, neutro ou negativo"
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        ajustar_barras(fig)
        fig.update_layout(
            xaxis_title="Aparelho mais usado nas redes sociais",
            yaxis_title="% das pessoas do grupo",
            yaxis_range=[0, 112],
            yaxis_ticksuffix="%",
            legend_title_text="Acham o impacto",
            font_size=14
        )

        st.plotly_chart(fig, use_container_width=True)

        beneficos = (
            longo[longo["Percepção"] == "Benéfico"]
            .sort_values("Porcentagem", ascending=False)
        )
        if len(beneficos) >= 2:
            topo, base = beneficos.iloc[0], beneficos.iloc[-1]
            dif = topo["Porcentagem"] - base["Porcentagem"]
            resumo = (
                f"👉 **Em resumo:** a maior parcela que acha o impacto "
                f"**benéfico** usa **{topo['Device_Type']}** "
                f"({porcentagem_br(topo['Porcentagem'])}) e a menor usa "
                f"**{base['Device_Type']}** "
                f"({porcentagem_br(base['Porcentagem'])}) — uma diferença "
                f"de **{numero_br(dif)} pontos percentuais**."
            )
            if dif < 5:
                resumo += (
                    " É uma diferença pequena: o aparelho quase não muda "
                    "a opinião das pessoas."
                )
            v_apar = cramer_v(d5["Device_Type"], d5["Overall_Impact"])
            if pd.notna(v_apar):
                resumo += (
                    f"  \n📈 **Associação aparelho × opinião** "
                    f"(V de Cramér): {numero_br(v_apar, 2)} — "
                    f"{forca_correlacao(v_apar)} "
                    f"(0 = nenhuma relação, 1 = relação total)."
                )
            st.info(resumo)

        st.caption(
            "Cada grupo de barras soma 100%. Grupos pequenos oscilam mais, "
            "por isso a quantidade de pessoas aparece embaixo do nome. "
            "Mostra uma associação, não uma relação de causa e efeito."
        )
    mostrar_relatorio(rel5)

# =====================================================================
# 6º GRÁFICO — Percepção do impacto por faixa etária e gênero
# =====================================================================

st.subheader("👥 Idade e gênero: a percepção do impacto muda?")

percepcao = st.radio(
    "Qual opinião você quer acompanhar?",
    ORDEM_IMPACTO,
    horizontal=True,
    format_func=lambda x: {
        "Benéfico": "Acham benéfico",
        "Neutro": "Acham neutro",
        "Negativo": "Acham negativo"
    }.get(x, x)
)

# Idade não passa por remoção de outliers: idades mais altas são reais
d6, rel6 = tratar(
    filtrado, ["Age"], ["Gender", "Overall_Impact"], outliers=False
)

if not sem_dados(d6, minimo=MIN_N):

    contagem_genero = d6["Gender"].value_counts()
    total6 = contagem_genero.sum()
    mostrar_cartoes([
        (
            g,
            f"{inteiro_br(contagem_genero[g])} pessoas",
            f"{porcentagem_br(contagem_genero[g] / total6 * 100)} do total",
            MAPA_COR_GENERO.get(g, "#777777")
        )
        for g in ORDEM_GENERO if g in contagem_genero.index
    ])

    d6["Faixa etária"] = pd.cut(
        d6["Age"],
        bins=[0, 17, 19, 21, float("inf")],
        labels=["Até 17 anos", "18 a 19 anos", "20 a 21 anos",
                "22 anos ou mais"]
    )
    d6["alvo"] = d6["Overall_Impact"] == percepcao

    grupos = (
        d6.groupby(["Faixa etária", "Gender"], observed=True)
        .agg(Porcentagem=("alvo", "mean"), Pessoas=("alvo", "size"))
        .reset_index()
    )
    grupos["Porcentagem"] = grupos["Porcentagem"] * 100
    grupos["Faixa etária"] = grupos["Faixa etária"].astype(str)

    pequenos = grupos[grupos["Pessoas"] < MIN_N]
    grupos = grupos[grupos["Pessoas"] >= MIN_N].copy()

    if not sem_dados(grupos, minimo=2):
        grupos["Texto"] = grupos["Porcentagem"].map(porcentagem_br)
        grupos = grupos.rename(columns={"Gender": "Gênero"})

        fig = px.bar(
            grupos,
            x="Faixa etária",
            y="Porcentagem",
            color="Gênero",
            color_discrete_map=MAPA_COR_GENERO,
            pattern_shape="Gênero" if PADROES_ATIVOS else None,
            pattern_shape_map=MAPA_PADRAO_GENERO,
            category_orders={
                "Faixa etária": ["Até 17 anos", "18 a 19 anos",
                                 "20 a 21 anos", "22 anos ou mais"],
                "Gênero": ORDEM_GENERO
            },
            text="Texto",
            barmode="group",
            hover_data={"Pessoas": True},
            title=f"% de pessoas que acham o impacto "
                  f"{percepcao.lower()}, por idade e gênero"
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        ajustar_barras(fig)
        fig.update_layout(
            xaxis_title="Faixa etária",
            yaxis_title="% das pessoas do grupo",
            yaxis_range=[0, max(25, grupos["Porcentagem"].max() * 1.2)],
            yaxis_ticksuffix="%",
            font_size=14
        )

        st.plotly_chart(fig, use_container_width=True)

        maior = grupos.loc[grupos["Porcentagem"].idxmax()]
        menor = grupos.loc[grupos["Porcentagem"].idxmin()]
        dif = maior["Porcentagem"] - menor["Porcentagem"]
        resumo = (
            f"👉 **Em resumo:** entre os grupos mostrados, a opinião "
            f"“{percepcao.lower()}” vai de "
            f"**{porcentagem_br(menor['Porcentagem'])}** "
            f"({menor['Gênero'].lower()}, {menor['Faixa etária'].lower()}) "
            f"a **{porcentagem_br(maior['Porcentagem'])}** "
            f"({maior['Gênero'].lower()}, {maior['Faixa etária'].lower()}) "
            f"— diferença de **{numero_br(dif)} pontos percentuais**."
        )
        if dif < 5:
            resumo += (
                " É pouco: idade e gênero quase não mudam a opinião."
            )
        menor_grupo = min(maior["Pessoas"], menor["Pessoas"])
        if menor_grupo < 100:
            resumo += (
                f" ⚠️ Cuidado: um desses dois grupos tem só "
                f"{inteiro_br(menor_grupo)} pessoas, então a porcentagem "
                f"dele oscila bastante e a diferença pode ser acaso."
            )
        rho_idade = d6["Age"].corr(d6["Impact_Score"], method="spearman")
        v_gen = cramer_v(d6["Gender"], d6["Overall_Impact"])
        if pd.notna(rho_idade) and pd.notna(v_gen):
            resumo += (
                f"  \n📈 **Idade × opinião** (Spearman): "
                f"{sinal_br(rho_idade)} — {forca_correlacao(rho_idade)}. "
                f"**Gênero × opinião** (V de Cramér): "
                f"{numero_br(v_gen, 2)} — {forca_correlacao(v_gen)}."
            )
        st.info(resumo)

        st.caption(
            "Cada grupo de barras compara os gêneros dentro de uma mesma "
            "faixa etária. Mostra uma associação, não uma relação de "
            "causa e efeito."
        )
        if not pequenos.empty:
            nomes = [
                f"{r['Gender'].lower()} ({r['Faixa etária'].lower()})"
                for _, r in pequenos.iterrows()
            ]
            st.caption(
                f"Grupos com menos de {MIN_N} pessoas foram omitidos para "
                f"não induzir a erro: {', '.join(nomes)}."
            )
    mostrar_relatorio(rel6)

# =====================================================================
# 7º GRÁFICO — Mapa de correlações
# =====================================================================

CORR_VARIAVEIS = {
    "Horas de uso por dia": "Daily_Usage_Hours",
    "Horas extras no fim de semana": "Weekend_Extra_Hours",
    "Horas de sono": "Sleep_Duration_Hours",
    "Qualidade do sono": "Sleep_Quality_Score",
    "Estresse": "Perceived_Stress_Score",
    "Saúde mental": "Mental_Health_Index",
    "Nota acadêmica": "Academic_Performance_GPA",
    "Idade": "Age",
    "Percepção do impacto": "Impact_Score",
}

st.subheader("🔗 Quais coisas andam juntas? (correlações)")
st.markdown(
    "A **correlação** mede se duas coisas variam juntas. Ela vai de "
    "**−1 a +1**: perto de **+1**, quando uma sobe a outra também sobe; "
    "perto de **−1**, quando uma sobe a outra desce; perto de **0**, uma "
    "não tem relação com a outra. Cada quadrado mostra o número e a cor "
    "mostra a força."
)

metodo = st.radio(
    "Tipo de correlação",
    ["Pearson", "Spearman"],
    horizontal=True,
    format_func=lambda m: {
        "Pearson": "Pearson (relação em linha reta)",
        "Spearman": "Spearman (relação de ordem, resiste a valores extremos)"
    }[m]
)

colunas_corr = list(CORR_VARIAVEIS.values())
nomes_corr = list(CORR_VARIAVEIS.keys())

# Idade e percepção ficam de fora da detecção de outliers: idades altas são
# reais e a percepção só tem 3 níveis (o IQR trataria "negativo" como outlier)
d7c, rel7c = tratar(
    filtrado, colunas_corr, sem_outliers=["Age", "Impact_Score"]
)

if not sem_dados(d7c, minimo=MIN_N):

    matriz = d7c[colunas_corr].corr(method=metodo.lower())
    matriz.index = nomes_corr
    matriz.columns = nomes_corr

    # Só o triângulo de baixo: cada par aparece uma única vez
    mascara = np.triu(np.ones(matriz.shape, dtype=bool), k=1)
    visivel = matriz.mask(mascara)
    textos = visivel.apply(
        lambda col: col.map(lambda v: "" if pd.isna(v) else sinal_br(v))
    )

    fig = px.imshow(
        visivel,
        zmin=-1,
        zmax=1,
        color_continuous_scale=MODO["divergente"],
        aspect="auto",
        title=f"Correlação de {metodo} entre as variáveis"
    )
    fig.update_traces(
        text=textos.values,
        texttemplate="%{text}",
        hovertemplate="%{y} × %{x}<br>Correlação: %{z:.2f}<extra></extra>"
    )
    fig.update_layout(
        height=640,
        font_size=13,
        xaxis_tickangle=-35,
        xaxis_title=None,
        yaxis_title=None,
        coloraxis_colorbar=dict(title="Correlação")
    )

    st.plotly_chart(fig, use_container_width=True)

    pares = [
        (nomes_corr[i], nomes_corr[j], matriz.iloc[i, j])
        for i in range(len(nomes_corr)) for j in range(i)
        if pd.notna(matriz.iloc[i, j])
    ]
    pares.sort(key=lambda p: abs(p[2]), reverse=True)

    if pares:
        linhas = [
            f"- **{a}** e **{b}**: {sinal_br(v)} "
            f"({forca_correlacao(v)}, {sentido_correlacao(v)})"
            for a, b, v in pares[:5]
        ]
        st.info(
            "👉 **As 5 relações mais fortes:**\n\n" + "\n".join(linhas)
        )

    st.caption(
        "Correlação não é causa: duas coisas podem andar juntas sem que "
        "uma provoque a outra. A “percepção do impacto” foi convertida em "
        "0 (negativo), 1 (neutro) e 2 (benéfico) para entrar na conta; "
        "para ela, o Spearman é o mais indicado."
    )
mostrar_relatorio(rel7c)

# =====================================================================
# 8º GRÁFICO — Análise multidimensional 3D
# =====================================================================

st.divider()

st.header("🚀 Análise Multidimensional 3D")

st.markdown(
    """
    Este gráfico reúne cinco variáveis em uma única visualização
    tridimensional. Cada esfera representa um participante,
    permitindo analisar simultaneamente diferentes aspectos
    relacionados ao uso das redes sociais.
    """
)

d7, rel7 = tratar(
    filtrado,
    [
        "Daily_Usage_Hours",
        "Sleep_Duration_Hours",
        "Mental_Health_Index",
        "Perceived_Stress_Score",
        "Academic_Performance_GPA"
    ],
    ["Overall_Impact", "Primary_Platform", "Gender"]
)

if not d7.empty:

    fig_3d = px.scatter_3d(
        d7,
        x="Daily_Usage_Hours",
        y="Sleep_Duration_Hours",
        z="Mental_Health_Index",
        color="Perceived_Stress_Score",
        size="Academic_Performance_GPA",
        size_max=12,
        symbol="Overall_Impact" if PADROES_ATIVOS else None,
        symbol_map=SIMBOLOS_IMPACTO,
        color_continuous_scale=MODO["escala"],
        hover_name="Primary_Platform",
        hover_data=[
            "Age",
            "Gender",
            "Overall_Impact",
            "Sleep_Quality_Score",
            "Academic_Performance_GPA"
        ],
        title="Uso de redes sociais × sono × saúde mental",
        labels={
            "Daily_Usage_Hours": "Uso diário (horas)",
            "Sleep_Duration_Hours": "Sono (horas)",
            "Mental_Health_Index": "Índice de saúde mental",
            "Perceived_Stress_Score": "Estresse percebido",
            "Academic_Performance_GPA": "Desempenho acadêmico"
        }
    )

    fig_3d.update_traces(
        marker=dict(
            opacity=0.8,
            line=dict(width=0.3, color=MODO["contorno"])
        )
    )

    fig_3d.update_layout(
        height=750,
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title="Uso diário (horas)",
            yaxis_title="Duração do sono (horas)",
            zaxis_title="Saúde mental",
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2)
            ),
            aspectmode="cube"
        ),
        coloraxis_colorbar=dict(
            title="Estresse<br>percebido"
        )
    )

    st.plotly_chart(
        fig_3d,
        use_container_width=True
    )

    st.markdown(
        """
        **Como interpretar o gráfico:**

        - O eixo X representa as horas diárias de uso das redes sociais.
        - O eixo Y representa a duração do sono.
        - O eixo Z representa o índice de saúde mental.
        - As cores representam o nível de estresse percebido.
        - O tamanho das esferas representa o desempenho acadêmico (GPA).
        - Com "padrões e símbolos" ativado, o formato do ponto indica o
          impacto percebido (círculo = benéfico, quadrado = neutro,
          losango = negativo).

        Cada ponto representa um participante. A visualização
        permite identificar agrupamentos e possíveis padrões
        entre os indicadores analisados.

        É importante destacar que o gráfico mostra associações
        entre variáveis, não relações comprovadas de causa e efeito.
        """
    )
    mostrar_relatorio(rel7)
    st.caption(
        "Como este gráfico usa cinco variáveis ao mesmo tempo, um registro "
        "é descartado se qualquer uma delas estiver ausente ou for outlier."
    )

else:
    st.warning(
        "Não existem registros completos suficientes "
        "para construir o gráfico 3D com os filtros atuais."
    )

# =====================================================================
# TABELA DE DADOS
# =====================================================================

st.divider()

with st.expander("📋 Visualizar dados completos"):

    st.write(
        f"Quantidade de participantes: {len(filtrado)}"
    )
    st.caption(
        "Mostra os dados após a limpeza da carga (duplicados e valores "
        "impossíveis) e dos filtros, sem o tratamento específico de "
        "cada gráfico."
    )

    tabela_visivel = filtrado.drop(columns=["Impact_Score"])

    st.dataframe(
        tabela_visivel,
        use_container_width=True,
        hide_index=True
    )

    csv = tabela_visivel.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Baixar dados filtrados",
        csv,
        "redes_sociais_filtrado.csv",
        "text/csv"
    )

st.caption(
    "Fonte: banco de dados O Impacto das Redes Sociais na Vida."
)
