# 0. Bibliotecas (pacotes)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score

print("==================================================================")
print("INICIANDO SCRIPT DE ANÁLISE DE CHURN EM PYTHON")
print("==================================================================")

# 3.1. Leitura e Limpeza Inicial dos Dados
print("\n>>> Etapa 3.1: Lendo e tratando os dados iniciais...")
# Para este exemplo, vou usar uma URL pública com o mesmo dataset.
# Se tiver o arquivo localmente, substitua a 'url' pelo caminho do arquivo.
# Ex: file_path = "e:/Newfolder0/ADGN/Telco-Customer-Churn.csv"
url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
dados = pd.read_csv(url)

# Converter TotalCharges para numérico. 'coerce' transforma erros em NaN.
dados['TotalCharges'] = pd.to_numeric(dados['TotalCharges'], errors='coerce')
# Substituir os valores nulos (NaN) por 0.
dados['TotalCharges'].fillna(0, inplace=True)

print("Dados lidos e tratados com sucesso.")

# 3.2. Seleção de Variáveis para a Análise
print("\n>>> Etapa 3.2: Selecionando as variáveis para o estudo...")
colunas_selecionadas = [
    'customerID', 'Churn', 'tenure', 'MonthlyCharges', 'TotalCharges',
    'Contract', 'InternetService', 'PaymentMethod'
]
dados_modelo = dados[colunas_selecionadas].copy()
print(f"Base de dados 'dados_modelo' criada com {dados_modelo.shape[1]} variáveis.")

# 3.3. Análise Exploratória - Gráficos (Tarefa 2)
print("\n>>> Etapa 3.3: Gerando gráficos boxplot (Tarefa 2)...")
plt.figure(figsize=(8, 6))
sns.boxplot(x='Churn', y='MonthlyCharges', data=dados_modelo)
plt.title('Boxplot de Cobranças Mensais por Churn')
plt.savefig('boxplot_churn_monthly_charges.png')

plt.figure(figsize=(8, 6))
sns.boxplot(x='Churn', y='TotalCharges', data=dados_modelo)
plt.title('Boxplot de Cobranças Totais por Churn')
plt.savefig('boxplot_churn_total_charges.png')

plt.figure(figsize=(8, 6))
sns.boxplot(x='Churn', y='tenure', data=dados_modelo)
plt.title('Boxplot de Tempo de Contrato (Tenure) por Churn')
plt.savefig('boxplot_churn_tenure.png')
print("Gráficos salvos na pasta atual.")


# 3.4. Preparação para Modelagem (Tarefa 3.a)
print("\n>>> Etapa 3.4: Particionando a base em Treino e Teste...")
treino, teste = train_test_split(
    dados_modelo,
    test_size=0.30,
    random_state=11299828,
    stratify=dados_modelo['Churn']
)
print(f"Base dividida: 70% para treino ({len(treino)} linhas) e 30% para teste ({len(teste)} linhas).")


print("\n==================================================================")
print("ANÁLISE DESCRITIVA DA BASE DE TREINO (TAREFA 3.a)")
print("==================================================================")

# --- Tabela para Variáveis Numéricas ---
print("\n--- Análise de Variáveis Numéricas por grupo de Churn ---")
tabela_numerica = treino.groupby('Churn')[['tenure', 'MonthlyCharges', 'TotalCharges']].agg(['mean', 'std'])
print(round(tabela_numerica, 2))

# --- Tabelas para Variáveis Categóricas ---
print("\n\n--- Análise de Variáveis Categóricas (Frequências) ---")
def criar_tabela_frequencia(data, nome_variavel):
    print(f"\n> Variável: {nome_variavel}")
    freq_abs = data[nome_variavel].value_counts()
    freq_rel = data[nome_variavel].value_counts(normalize=True).map('{:.1%}'.format)
    tabela = pd.DataFrame({'Freq. Absoluta': freq_abs, 'Freq. Relativa': freq_rel})
    print(tabela)

criar_tabela_frequencia(treino, "Contract")
criar_tabela_frequencia(treino, "InternetService")
criar_tabela_frequencia(treino, "PaymentMethod")


# 4. Modelagem
print("\n==================================================================")
print("INICIANDO ETAPA DE MODELAGEM (TAREFA 3.b)")
print("==================================================================")

# --- Preparação Final para Modelagem ---
X_treino = treino.drop(columns=['customerID', 'Churn'])
y_treino = treino['Churn']
X_teste = teste.drop(columns=['customerID', 'Churn'])
y_teste = teste['Churn']

numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['Contract', 'InternetService', 'PaymentMethod']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- Modelo 1: Regressão Logística (glm) ---
print("\n--- 1. Treinando Modelo: Regressão Logística ---")
pipeline_log = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', LogisticRegression(random_state=11299828))])
pipeline_log.fit(X_treino, y_treino)
print("Modelo 'pipeline_log' treinado.")

# --- Modelo 2: Random Forest (rf) ---
print("\n--- 2. Treinando Modelo: Random Forest ---")
pipeline_rf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=11299828))])
pipeline_rf.fit(X_treino, y_treino)
print("Modelo 'pipeline_rf' treinado.")

# --- Modelo 3: K-Nearest Neighbors (k-NN) ---
print("\n--- 3. Treinando Modelo: K-Nearest Neighbors ---")
pipeline_knn = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', KNeighborsClassifier())])
pipeline_knn.fit(X_treino, y_treino)
print("Modelo 'pipeline_knn' treinado.")


# 5. Avaliação de Performance e Variância dos Modelos
print("\n==================================================================")
print("AVALIAÇÃO DE PERFORMANCE DOS MODELOS (TAREFA 3.b.ii e 3.b.iii)")
print("==================================================================")

modelos = {
    "Regressão Logística": pipeline_log,
    "Random Forest": pipeline_rf,
    "k-NN": pipeline_knn
}

resultados = []

for nome, modelo in modelos.items():
    print(f"\n==================== ANÁLISE DO MODELO: {nome} ====================")
    
    # Previsões
    pred_treino = modelo.predict(X_treino)
    pred_teste = modelo.predict(X_teste)

    # --- Matriz de Confusão (TREINO) ---
    print(f"\n--- Matriz de Confusão (TREINO) para o modelo: {nome} ---")
    cm_treino = confusion_matrix(y_treino, pred_treino, labels=modelo.classes_)
    print(cm_treino)

    # --- Matriz de Confusão (TESTE) ---
    print(f"\n--- Matriz de Confusão (TESTE) para o modelo: {nome} ---")
    cm_teste = confusion_matrix(y_teste, pred_teste, labels=modelo.classes_)
    print(cm_teste)
    print(classification_report(y_teste, pred_teste))

    # Métricas para a tabela resumo
    acc_treino = accuracy_score(y_treino, pred_treino)
    f1_treino  = f1_score(y_treino, pred_treino, pos_label='Yes')
    acc_teste  = accuracy_score(y_teste, pred_teste)
    f1_teste   = f1_score(y_teste, pred_teste, pos_label='Yes')
    
    resultados.append({
        "Modelo": nome,
        "Acuracia_Treino": acc_treino,
        "F1_Score_Treino": f1_treino,
        "Acuracia_Teste": acc_teste,
        "F1_Score_Teste": f1_teste
    })

# --- Tabela Resumo e Análise de Variância ---
tabela_resumo = pd.DataFrame(resultados)
tabela_resumo['Variancia_Acuracia'] = tabela_resumo['Acuracia_Treino'] - tabela_resumo['Acuracia_Teste']
tabela_resumo['Variancia_F1_Score'] = tabela_resumo['F1_Score_Treino'] - tabela_resumo['F1_Score_Teste']

print("\n\n--- Tabela Resumo de Métricas e Análise de Variância ---\n")
print(tabela_resumo.round(4))

print("\n==================================================================")
print("SCRIPT FINALIZADO")
print("==================================================================")