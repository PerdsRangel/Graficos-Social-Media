
import streamlit as st
import pandas as pd
import plotly.express as px

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

# ==========================================
# CARREGAMENTO DOS DADOS
# ==========================================

@st.cache_data
def carregar_dados():
    df = pd.read_csv("Social_media_impact_on_life.csv")

    df.columns = df.columns.str.strip()

    colunas_numericas = [
        "Age",
        "Daily_Usage_Hours",
        "Weekend_Extra_Hours",
        "Sleep_Duration_Hours",
        "Sleep_Quality_Score",
        "Perceived_Stress_Score",
        "Mental_Health_Index",
        "Academic_Performance_GPA"
    ]

    for coluna in colunas_numericas:
        df[coluna] = pd.to_numeric(
            df[coluna], errors="coerce"
        )

    return df


df = carregar_dados()

# ==========================================
# FILTROS
# ==========================================

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
genero = st.sidebar.multiselect(
    "Gênero",
    generos,
    default=generos
)

plataformas = sorted(df["Primary_Platform"].dropna().unique())
plataforma = st.sidebar.multiselect(
    "Plataforma principal",
    plataformas,
    default=plataformas
)

niveis = sorted(df["Academic_Level"].dropna().unique())
nivel = st.sidebar.multiselect(
    "Nível acadêmico",
    niveis,
    default=niveis
)

impactos = sorted(df["Overall_Impact"].dropna().unique())
impacto = st.sidebar.multiselect(
    "Impacto percebido",
    impactos,
    default=impactos
)

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

# ==========================================
# INDICADORES PRINCIPAIS
# ==========================================

st.subheader("📊 Indicadores gerais")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Participantes", f"{len(filtrado):,}")
c2.metric(
    "Uso diário médio",
    f"{filtrado['Daily_Usage_Hours'].mean():.1f} h"
)
c3.metric(
    "Sono médio",
    f"{filtrado['Sleep_Duration_Hours'].mean():.1f} h"
)
c4.metric(
    "Estresse médio",
    f"{filtrado['Perceived_Stress_Score'].mean():.1f}"
)
c5.metric(
    "Saúde mental",
    f"{filtrado['Mental_Health_Index'].mean():.1f}"
)

st.caption(
    "Os indicadores apresentam as médias dos participantes "
    "considerados após a aplicação dos filtros."
)

st.divider()

# ==========================================
# GRÁFICO 1 - PLATAFORMAS
# ==========================================

st.subheader("📱 Plataformas mais utilizadas")

dados = (
    filtrado["Primary_Platform"]
    .value_counts()
    .rename_axis("Plataforma")
    .reset_index(name="Participantes")
)

fig = px.bar(
    dados,
    x="Plataforma",
    y="Participantes",
    color="Plataforma",
    title="Distribuição das plataformas utilizadas"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Este gráfico mostra a quantidade de participantes que "
    "utilizam cada plataforma como sua principal rede social. "
    "As maiores barras representam as plataformas mais populares "
    "entre os participantes."
)

# ==========================================
# GRÁFICO 2 - IMPACTO GERAL
# ==========================================

st.subheader("🌐 Percepção do impacto das redes sociais")

dados = (
    filtrado["Overall_Impact"]
    .value_counts()
    .rename_axis("Impacto")
    .reset_index(name="Participantes")
)

fig = px.pie(
    dados,
    names="Impacto",
    values="Participantes",
    hole=0.45,
    title="Percepção geral dos participantes"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "O gráfico apresenta a proporção de participantes que "
    "consideram o impacto das redes sociais positivo, neutro "
    "ou negativo. Ele permite visualizar a percepção geral "
    "dos entrevistados."
)

# ==========================================
# GRÁFICO 3 - TEMPO DE USO
# ==========================================

st.subheader("⏱️ Distribuição do tempo diário de uso")

fig = px.histogram(
    filtrado,
    x="Daily_Usage_Hours",
    nbins=25,
    color_discrete_sequence=["#636EFA"],
    title="Horas diárias nas redes sociais"
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

# ==========================================
# GRÁFICO 4 - USO X SONO
# ==========================================

st.subheader("😴 Relação entre uso das redes sociais e sono")

fig = px.scatter(
    filtrado,
    x="Daily_Usage_Hours",
    y="Sleep_Duration_Hours",
    color="Overall_Impact",
    hover_data=["Age", "Primary_Platform"],
    title="Tempo de uso diário versus duração do sono",
    opacity=0.7
)

fig.update_layout(
    xaxis_title="Uso diário (horas)",
    yaxis_title="Duração do sono (horas)"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Cada ponto representa um participante. O gráfico permite "
    "observar se existe algum padrão entre o tempo de utilização "
    "das redes sociais e a duração do sono. Uma associação visual "
    "não significa necessariamente uma relação de causa e efeito."
)

# ==========================================
# GRÁFICO 5 - COMPARAÇÃO POR PLATAFORMA
# ==========================================

st.subheader("📊 Comparação entre plataformas")

dados = (
    filtrado.groupby("Primary_Platform", as_index=False)
    .agg(
        Uso_diario=("Daily_Usage_Hours", "mean"),
        Sono=("Sleep_Duration_Hours", "mean")
    )
)

fig = px.bar(
    dados,
    x="Primary_Platform",
    y=["Uso_diario", "Sono"],
    barmode="group",
    title="Média de uso e sono por plataforma"
)

fig.update_layout(
    xaxis_title="Plataforma",
    yaxis_title="Média em horas"
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Compara as médias de horas de uso diário e duração do sono "
    "entre as diferentes plataformas. Isso permite identificar "
    "diferenças nos hábitos dos grupos analisados."
)

# ==========================================
# GRÁFICO 6 - RELAÇÃO COM OUTROS INDICADORES
# ==========================================

st.subheader("🧠 Relação entre uso e indicadores")

indicadores = {
    "Duração do sono": "Sleep_Duration_Hours",
    "Qualidade do sono": "Sleep_Quality_Score",
    "Estresse percebido": "Perceived_Stress_Score",
    "Índice de saúde mental": "Mental_Health_Index",
    "Desempenho acadêmico (GPA)": "Academic_Performance_GPA"
}

nome_indicador = st.selectbox(
    "Escolha o indicador para analisar:",
    list(indicadores.keys())
)

coluna = indicadores[nome_indicador]

relacao = filtrado.dropna(
    subset=["Daily_Usage_Hours", coluna]
)

fig = px.scatter(
    relacao,
    x="Daily_Usage_Hours",
    y=coluna,
    trendline="ols",
    opacity=0.65,
    title=f"Uso diário versus {nome_indicador}"
)

fig.update_layout(
    xaxis_title="Horas de uso diário",
    yaxis_title=nome_indicador
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Este gráfico apresenta a relação entre o tempo de uso "
    "das redes sociais e o indicador selecionado. A linha "
    "de tendência resume o comportamento geral dos dados. "
    "A associação observada não comprova causalidade."
)

# ==========================================
# GRÁFICO 7 - ANÁLISE MULTIDIMENSIONAL 3D
# ==========================================

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

dados_3d = filtrado.dropna(
    subset=[
        "Daily_Usage_Hours",
        "Sleep_Duration_Hours",
        "Mental_Health_Index",
        "Perceived_Stress_Score",
        "Academic_Performance_GPA"
    ]
).copy()

if not dados_3d.empty:

    fig_3d = px.scatter_3d(
        dados_3d,
        x="Daily_Usage_Hours",
        y="Sleep_Duration_Hours",
        z="Mental_Health_Index",
        color="Perceived_Stress_Score",
        size="Academic_Performance_GPA",
        size_max=12,
        color_continuous_scale="Turbo",
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
            opacity=0.75,
            line=dict(width=0.3, color="white")
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

        Cada ponto representa um participante. A visualização
        permite identificar agrupamentos e possíveis padrões
        entre os indicadores analisados.

        É importante destacar que o gráfico mostra associações
        entre variáveis, não relações comprovadas de causa e efeito.
        """
    )

else:
    st.warning(
        "Não existem registros completos suficientes "
        "para construir o gráfico 3D com os filtros atuais."
    )

# ==========================================
# TABELA DE DADOS
# ==========================================

st.divider()

with st.expander("📋 Visualizar dados completos"):

    st.write(
        f"Quantidade de participantes: {len(filtrado)}"
    )

    st.dataframe(
        filtrado,
        use_container_width=True,
        hide_index=True
    )

    csv = filtrado.to_csv(
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