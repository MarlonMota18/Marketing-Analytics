# Análise e Previsão de Churn de Clientes (Telecom) 📊

## 📖 Sobre o Projeto

Este repositório contém o desenvolvimento de uma análise de dados voltada para *Marketing Analytics*, desenvolvida como parte da disciplina RAD2810 - Análise de Dados para Gestão de Negócios (USP).

O objetivo principal é entender o comportamento de clientes de uma empresa de telecomunicações e construir modelos de *Machine Learning* capazes de prever o *churn* (cancelamento de serviços). Para empresas que operam com modelos de assinatura, prever a evasão de clientes é vital para direcionar campanhas de retenção e garantir a saúde financeira do negócio.

## 🛠️ Stack Tecnológico

* **Linguagem:** Python
* **Manipulação e Análise de Dados:** Pandas, NumPy
* **Visualização:** Seaborn, Matplotlib
* **Machine Learning:** Scikit-Learn

## 🗂️ Conjunto de Dados e Tratamento

A base utilizada foi a `Telco-Customer-Churn.csv`. A variável alvo (*target*) do modelo é a coluna `Churn` (0 = Retido, 1 = Cancelado).

Antes da modelagem, o *dataset* passou por etapas de limpeza, como a coerção de tipos da variável `TotalCharges` e o tratamento de valores nulos (imputação). Foram selecionadas para o escopo preditivo as variáveis de maior relevância de negócio:

* **Numéricas:** `tenure` (meses de permanência), `MonthlyCharges` (cobrança mensal), `TotalCharges` (cobrança total).
* **Categóricas:** `Contract` (tipo de contrato), `InternetService` (tipo de internet), `PaymentMethod` (método de pagamento).

## ⚙️ Metodologia Aplicada

### 1. Análise Exploratória de Dados (EDA)

Por meio de estatísticas descritivas e visualizações (*Boxplots*), identificamos que a variável `tenure` é um forte preditor de churn. A grande maioria dos cancelamentos ocorre no início do ciclo de vida do cliente, indicando a necessidade de estratégias de engajamento (*onboarding*) mais fortes nos primeiros meses.

### 2. Pré-processamento e Pipelines

Para garantir boas práticas de desenvolvimento e evitar o vazamento de dados (*data leakage*), o pré-processamento foi estruturado utilizando `Pipeline` e `ColumnTransformer` do Scikit-Learn:

* **Variáveis Numéricas:** Padronizadas utilizando `StandardScaler` para otimizar a convergência dos algoritmos.
* **Variáveis Categóricas:** Transformadas via `OneHotEncoder` para permitir a leitura matemática pelos modelos.

### 3. Modelagem Preditiva

Os dados foram divididos em **70% para treino e 30% para teste**, aplicando o parâmetro de **estratificação** (`stratify`) na variável alvo para garantir a mesma proporção de classes nas duas bases. Foram treinados e testados três classificadores:

1. **Regressão Logística**
2. **Random Forest**
3. **K-Nearest Neighbors (KNN)**

## 📈 Avaliação de Desempenho

Os modelos foram validados pelas métricas de **Acurácia**, **F1-Score** (essencial para bases onde falsos positivos e falsos negativos têm pesos de negócio diferentes) e **Variância** entre as bases de treino e teste.

| Modelo | Acurácia (Teste) | F1-Score (Teste) | Variância | Status de Overfitting |
| --- | --- | --- | --- | --- |
| **Regressão Logística** | **80,31%** | **0,5906** | **-1,02%** | Mínimo / Inexistente |
| **K-Nearest Neighbors** | 78,37% | 0,5668 | 5,73% | Baixo |
| **Random Forest** | 77,76% | 0,5455 | 21,61% | **Alto (Forte Overfitting)** |

## 🏆 Conclusão e Recomendação de Negócio

Após a análise das matrizes de confusão e dos escores, a **Regressão Logística demonstrou ser a melhor opção técnica e de negócio**.

Apesar de o algoritmo de *Random Forest* apresentar métricas excelentes na base de treino (decorando os dados), ele falhou na generalização, resultando em uma variância de mais de 21% (*forte overfitting*), tornando-o inadequado para o mundo real.

A Regressão Logística entregou a maior acurácia (**80,3%**) e o melhor F1-Score (**0,59**), destacando-se pela extrema **estabilidade**. Sua capacidade de generalização garante que a empresa possa aplicar o modelo em novos clientes de forma confiável, identificando com precisão os usuários com alta propensão ao cancelamento e subsidiando ações de marketing direcionadas.
