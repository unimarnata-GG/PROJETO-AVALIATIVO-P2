# Classificação de Doença Hepática — Machine Learning com Streamlit

> Aplicação web interativa para classificação de pacientes com possível indicativo de doença hepática, desenvolvida com Python, Scikit-learn e Streamlit.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Numerical%20Computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-purple)]()

**Acesso ao app:** adicionar aqui o link gerado pelo Streamlit Community Cloud.

---

## Sumário

1. [Sobre o Projeto](#1-sobre-o-projeto)
2. [Objetivo](#2-objetivo)
3. [Sobre o Dataset](#3-sobre-o-dataset)
4. [Estrutura do Projeto](#4-estrutura-do-projeto)
5. [Variável Alvo](#5-variável-alvo)
6. [Variáveis Preditoras](#6-variáveis-preditoras)
7. [Metodologia](#7-metodologia)
8. [Modelos Treinados](#8-modelos-treinados)
9. [Melhor Modelo Selecionado](#9-melhor-modelo-selecionado)
10. [Métricas de Avaliação](#10-métricas-de-avaliação)
11. [Funcionamento da Aplicação](#11-funcionamento-da-aplicação)
12. [Tecnologias Utilizadas](#12-tecnologias-utilizadas)
13. [Como Executar o Projeto](#13-como-executar-o-projeto)
14. [Conclusão](#14-conclusão)

---

## 1. Sobre o Projeto

Este projeto aplica técnicas de aprendizado de máquina a dados clínicos do **Indian Liver Patient Records**, com o objetivo de classificar pacientes com ou sem possível indicativo de doença hepática. A proposta contempla o fluxo completo de ciência de dados: análise exploratória, tratamento dos dados, treinamento de classificadores, avaliação por métricas e disponibilização do melhor modelo em uma aplicação web com Streamlit.

A aplicação permite que um usuário informe atributos clínicos e laboratoriais, como idade, gênero, bilirrubina, enzimas hepáticas e proteínas, recebendo como saída uma classificação estimada pelo modelo.

> Observação: este projeto possui finalidade acadêmica e não deve ser usado como ferramenta de diagnóstico médico.

---

## 2. Objetivo

**Objetivo geral:** Desenvolver um modelo de classificação capaz de estimar se um paciente possui ou não possível indicativo de doença hepática a partir de atributos clínicos e laboratoriais.

**Objetivos específicos:**

- Realizar análise exploratória dos dados, incluindo distribuição das variáveis, análise por gênero, outliers e correlação;
- Treinar e comparar diferentes classificadores de machine learning;
- Avaliar os modelos com validação cruzada estratificada e métricas adequadas para classificação;
- Selecionar o melhor modelo com base no desempenho obtido;
- Serializar o modelo treinado e o normalizador para uso em produção;
- Implementar uma aplicação Streamlit para realizar inferências com novos dados.

---

## 3. Sobre o Dataset

O dataset utilizado foi o **Indian Liver Patient Records**, uma base pública contendo registros de pacientes com atributos clínicos relacionados à função hepática.

No projeto, o dataset é carregado no notebook e posteriormente salvo em sua versão tratada:

```text
data/dataset_tratado.csv
```

A base contém variáveis como idade, gênero, bilirrubina total e direta, fosfatase alcalina, enzimas hepáticas ALT e AST, proteína total, albumina e relação albumina/globulina.

A divisão adotada no notebook foi:

| Partição | Percentual |
|---|---:|
| Treinamento | 70% |
| Validação | 15% |
| Teste | 15% |

A separação foi realizada com `stratify=y`, preservando a proporção das classes em treino, validação e teste.

---

## 4. Estrutura do Projeto

```text
p2/
│
├── app.py                         # Aplicação Streamlit para inferência
├── requirements.txt               # Dependências do projeto
├── README.md                      # Documentação do projeto
│
├── data/
│   └── dataset_tratado.csv         # Dataset tratado gerado pelo notebook
│
├── model/
│   ├── modelo_final.joblib         # Melhor modelo treinado
│   ├── scaler.joblib               # StandardScaler usado na normalização
│   └── metadata.json               # Informações do modelo, features e métricas
│
├── notebooks/
│   └── notebook_atualizado.ipynb   # Notebook de análise, treinamento e salvamento
│
└── reports/
    └── relatorio_atualizado.pdf    # Relatório final do projeto
```

---

## 5. Variável Alvo

| Variável | Descrição | Tipo |
|---|---|---|
| `Classe` | Indica se o paciente possui ou não possível doença hepática | Binária |

Mapeamento utilizado:

| Valor | Interpretação |
|---:|---|
| `0` | Sem indicativo de doença hepática |
| `1` | Possível indicativo de doença hepática |

No tratamento dos dados, a classe original do dataset foi convertida para o padrão binário acima, facilitando a interpretação dos resultados e o uso no app.

---

## 6. Variáveis Preditoras

| Feature | Tipo | Descrição |
|---|---|---|
| `Idade` | Numérica | Idade do paciente |
| `Genero` | Binária | Gênero codificado como 0 para feminino e 1 para masculino |
| `Bilirrubina_Total` | Numérica | Quantidade total de bilirrubina no sangue |
| `Bilirrubina_Direta` | Numérica | Fração direta da bilirrubina |
| `Fosfatase_Alcalina` | Numérica | Enzima associada ao fígado e vias biliares |
| `ALT` | Numérica | Alanina aminotransferase, enzima relacionada à função hepática |
| `AST` | Numérica | Aspartato aminotransferase, enzima relacionada à função hepática |
| `Proteina_Total` | Numérica | Quantidade total de proteínas no sangue |
| `Albumina` | Numérica | Proteína produzida pelo fígado |
| `Relacao_A_G` | Numérica | Relação albumina/globulina |

---

## 7. Metodologia

```text
Dataset original
        │
        ▼
1. Carregamento dos dados
        │
        ▼
2. Renomeação das colunas
        │
        ▼
3. Tratamento de valores ausentes
        │
        ▼
4. Codificação de variáveis categóricas
   └── Gênero: feminino = 0, masculino = 1
        │
        ▼
5. Conversão da variável alvo
   └── Classe: 0 = sem indicativo, 1 = possível indicativo
        │
        ▼
6. Análise exploratória dos dados
   ├── Distribuição das variáveis
   ├── Análise de outliers
   ├── Correlação
   └── Análise por gênero
        │
        ▼
7. Divisão dos dados
   └── Treino 70%, validação 15%, teste 15%
        │
        ▼
8. Normalização com StandardScaler
        │
        ▼
9. Treinamento e comparação dos classificadores
        │
        ▼
10. Avaliação com Stratified K-Fold
        │
        ▼
11. Seleção do melhor modelo
        │
        ▼
12. Avaliação final no conjunto de teste
        │
        ▼
13. Salvamento do modelo, scaler e metadados
        │
        ▼
14. Uso do modelo na aplicação Streamlit
```

Esta versão mantém o fluxo sem `Pipeline`. Por isso, o `StandardScaler` é salvo separadamente e carregado no Streamlit antes da predição.

---

## 8. Modelos Treinados

| Algoritmo | Papel na Comparação |
|---|---|
| Logistic Regression | Modelo linear usado como baseline interpretável |
| Random Forest | Modelo baseado em ensemble de árvores por bagging |
| Gradient Boosting | Modelo baseado em ensemble sequencial por boosting |

Todos os modelos foram comparados usando validação cruzada estratificada e métricas de classificação.

---

## 9. Melhor Modelo Selecionado

O melhor modelo é selecionado automaticamente na última célula do notebook com base na métrica **AUC-ROC** calculada sobre os resultados de validação.

Após executar o notebook, preencher:

| Critério | Resultado |
|---|---|
| Melhor modelo selecionado | Preencher com o valor exibido no notebook |
| Critério de escolha | AUC-ROC |
| Arquivo do modelo | `model/modelo_final.joblib` |
| Arquivo do scaler | `model/scaler.joblib` |

O uso da AUC-ROC como critério é adequado porque ela avalia a capacidade geral do modelo de separar as duas classes, considerando diferentes limiares de decisão.

---

## 10. Métricas de Avaliação

As métricas calculadas no projeto foram:

| Métrica | O que mede | Como interpretar |
|---|---|---|
| Acurácia | Proporção geral de acertos | Quanto maior, melhor |
| Precisão | Entre os positivos previstos, quantos estavam corretos | Importante para reduzir falsos positivos |
| Recall | Entre os positivos reais, quantos foram encontrados | Importante para reduzir falsos negativos |
| F1-Score | Média harmônica entre precisão e recall | Útil quando há desequilíbrio entre classes |
| AUC-ROC | Capacidade de separação entre as classes | Quanto mais próximo de 1, melhor |

Após executar o notebook, preencher a tabela abaixo com os resultados finais do conjunto de teste:

| Métrica | Resultado |
|---|---:|
| Acurácia | preencher |
| Precisão | preencher |
| Recall | preencher |
| F1-Score | preencher |
| AUC-ROC | preencher |

---

## 11. Funcionamento da Aplicação

A aplicação Streamlit carrega os arquivos gerados pelo notebook:

```text
model/modelo_final.joblib
model/scaler.joblib
model/metadata.json
```

O usuário informa os dados clínicos no formulário da interface. Em seguida, o app:

1. Organiza os valores informados em um `DataFrame`;
2. Aplica o `StandardScaler` salvo em `scaler.joblib`;
3. Envia os dados normalizados para o modelo salvo;
4. Exibe a classe prevista;
5. Exibe a probabilidade estimada para a classe 1, quando disponível.

Adicionar aqui prints reais da aplicação funcionando:

```text
docs/prints/formulario.png
docs/prints/resultado.png
```

---

## 12. Tecnologias Utilizadas

| Categoria | Tecnologia | Uso |
|---|---|---|
| Linguagem | Python | Desenvolvimento do notebook e da aplicação |
| Interface web | Streamlit | Construção do app interativo |
| Machine Learning | Scikit-learn | Treinamento, avaliação e serialização dos modelos |
| Manipulação de dados | Pandas | Leitura, tratamento e organização dos dados |
| Computação numérica | NumPy | Operações numéricas |
| Visualização | Matplotlib e Seaborn | Gráficos da análise exploratória |
| Serialização | Joblib | Salvamento do modelo e scaler |

---

## 13. Como Executar o Projeto

**Pré-requisito:** Python 3.10 ou superior instalado.

```bash
# 1. Clone o repositório
git clone LINK_DO_REPOSITORIO
cd NOME_DO_REPOSITORIO

# 2. Crie e ative o ambiente virtual
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Execute a aplicação
streamlit run app.py
```

A aplicação estará disponível em:

```text
http://localhost:8501
```

Caso os arquivos em `model/` ainda não existam, execute o notebook abaixo até a última célula:

```text
notebooks/notebook_atualizado.ipynb
```

Essa execução gera:

```text
model/modelo_final.joblib
model/scaler.joblib
model/metadata.json
data/dataset_tratado.csv
```

---

## 14. Conclusão

O projeto atingiu o objetivo de construir um fluxo completo de classificação com machine learning, incluindo análise exploratória, treinamento de modelos, avaliação quantitativa, seleção do melhor classificador e disponibilização em uma aplicação web interativa.

Como limitação, o modelo foi treinado sobre uma base pública específica e deve ser interpretado apenas em contexto acadêmico, sem finalidade diagnóstica real. Além disso, a qualidade da classificação depende da representatividade dos dados disponíveis e das variáveis presentes no dataset.

Como trabalhos futuros, seria possível:

- Testar técnicas adicionais de balanceamento de classes;
- Realizar ajuste de hiperparâmetros com `GridSearchCV` ou `RandomizedSearchCV`;
- Adicionar explicabilidade com importância de variáveis ou SHAP;
- Melhorar a interface do Streamlit com histórico de predições e gráficos explicativos.

---

*Projeto desenvolvido como atividade avaliativa da disciplina de Inteligência Artificial.*
