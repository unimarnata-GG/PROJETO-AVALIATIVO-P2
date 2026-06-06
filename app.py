from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "modelo_final.joblib"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"


st.set_page_config(
    page_title="Classificação Hepática",
    page_icon="IA",
    layout="centered",
)


@st.cache_resource
def carregar_artefatos():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Arquivo model/modelo_final.joblib não encontrado. "
            "Execute o notebook até a última célula para gerar o pipeline final."
        )

    modelo = joblib.load(MODEL_PATH)

    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return modelo, metadata


def classificar_faixa(probabilidade):
    if probabilidade < 0.40:
        return "Baixa", "A probabilidade estimada ficou abaixo de 40%."
    if probabilidade < 0.70:
        return "Intermediária", "A probabilidade estimada ficou entre 40% e 70%."
    return "Alta", "A probabilidade estimada ficou acima de 70%."


def calcular_contribuicoes(modelo, entrada, features):
    if not hasattr(modelo, "named_steps"):
        return None

    scaler = modelo.named_steps.get("scaler")
    estimador = modelo.named_steps.get("model")

    if scaler is None or estimador is None or not hasattr(estimador, "coef_"):
        return None

    entrada_transformada = scaler.transform(entrada)
    coeficientes = estimador.coef_[0]
    contribuicoes = entrada_transformada[0] * coeficientes

    df_contribuicoes = pd.DataFrame(
        {
            "Variável": features,
            "Contribuição": contribuicoes,
            "Efeito": [
                "Aumenta chance da classe 1" if valor > 0 else "Reduz chance da classe 1"
                for valor in contribuicoes
            ],
        }
    )

    df_contribuicoes["Impacto absoluto"] = df_contribuicoes["Contribuição"].abs()
    return df_contribuicoes.sort_values("Impacto absoluto", ascending=False)


st.title("Classificação de Pacientes com Possível Doença Hepática")
st.caption(
    "Aplicação acadêmica baseada no dataset Indian Liver Patient Records. "
    "O resultado não substitui avaliação médica."
)

try:
    modelo, metadata = carregar_artefatos()
except FileNotFoundError as erro:
    st.error(str(erro))
    st.stop()


features = metadata.get(
    "features",
    [
        "Idade",
        "Genero",
        "Bilirrubina_Total",
        "Bilirrubina_Direta",
        "Fosfatase_Alcalina",
        "ALT",
        "AST",
        "Proteina_Total",
        "Albumina",
        "Relacao_A_G",
    ],
)

with st.form("formulario_predicao"):
    col1, col2 = st.columns(2)

    with col1:
        idade = st.number_input("Idade", min_value=0, max_value=120, value=45)
        genero_texto = st.selectbox("Gênero", ["Feminino", "Masculino"])
        bilirrubina_total = st.number_input("Bilirrubina total", min_value=0.0, value=1.0, step=0.1)
        bilirrubina_direta = st.number_input("Bilirrubina direta", min_value=0.0, value=0.3, step=0.1)
        fosfatase = st.number_input("Fosfatase alcalina", min_value=0.0, value=200.0, step=1.0)

    with col2:
        alt = st.number_input("ALT", min_value=0.0, value=30.0, step=1.0)
        ast = st.number_input("AST", min_value=0.0, value=35.0, step=1.0)
        proteina_total = st.number_input("Proteína total", min_value=0.0, value=6.5, step=0.1)
        albumina = st.number_input("Albumina", min_value=0.0, value=3.5, step=0.1)
        relacao_ag = st.number_input("Relação A/G", min_value=0.0, value=1.0, step=0.1)

    enviar = st.form_submit_button("Executar predição")


if enviar:
    genero = 1 if genero_texto == "Masculino" else 0

    entrada = pd.DataFrame(
        [
            {
                "Idade": idade,
                "Genero": genero,
                "Bilirrubina_Total": bilirrubina_total,
                "Bilirrubina_Direta": bilirrubina_direta,
                "Fosfatase_Alcalina": fosfatase,
                "ALT": alt,
                "AST": ast,
                "Proteina_Total": proteina_total,
                "Albumina": albumina,
                "Relacao_A_G": relacao_ag,
            }
        ]
    )

    entrada = entrada[features]

    # O arquivo modelo_final.joblib é um Pipeline completo.
    # Ele aplica o StandardScaler internamente antes da predição.
    predicao = int(modelo.predict(entrada)[0])
    probabilidade = None

    if hasattr(modelo, "predict_proba"):
        probabilidade = float(modelo.predict_proba(entrada)[0][1])

    if predicao == 1:
        st.error("Resultado: possível indicativo de doença hepática.")
    else:
        st.success("Resultado: sem indicativo de doença hepática.")

    if probabilidade is not None:
        faixa, descricao_faixa = classificar_faixa(probabilidade)
        st.metric("Probabilidade estimada para a classe 1", f"{probabilidade:.2%}")
        st.write(f"Faixa interpretativa: **{faixa}**. {descricao_faixa}")

    contribuicoes = calcular_contribuicoes(modelo, entrada, features)

    if contribuicoes is not None:
        st.subheader("Fatores que mais influenciaram a predição")
        st.caption(
            "Interpretação aproximada baseada nos coeficientes da Regressão Logística. "
            "Valores positivos aumentam a tendência para a classe 1; valores negativos reduzem essa tendência."
        )
        st.dataframe(
            contribuicoes[["Variável", "Contribuição", "Efeito"]].head(5),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "O modelo carregado não possui coeficientes lineares disponíveis para detalhar "
            "as contribuições individuais das variáveis."
        )

    with st.expander("Dados enviados ao modelo"):
        st.dataframe(entrada, use_container_width=True)


if metadata:
    st.divider()
    st.write("Modelo carregado:", metadata.get("melhor_modelo", "não informado"))
    st.write("Critério de escolha:", metadata.get("criterio_escolha", "não informado"))
