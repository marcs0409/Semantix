import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset

print("=== INICIANDO ROTINA DE MONITORAMENTO (EVIDENTLY AI) ===")

# 1. Carregar a base de dados
caminho_csv = "data/Churn_Modelling.csv"
if not os.path.exists(caminho_csv):
    raise FileNotFoundError(f"Arquivo '{caminho_csv}' não encontrado. Certifique-se de que a pasta /data está correta.")

df = pd.read_csv(caminho_csv, delimiter=',')

# Filtrar apenas as colunas preditoras de interesse
colunas_features = [
    'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
    'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
]
df_ml = df[colunas_features].copy()

# Tratamento básico de dados (Codificação de Gender para bater com o processamento do notebook)
df_ml['Gender'] = df_ml['Gender'].map({'Female': 0, 'Male': 1})

# 2. Dividir em dados de referência (histórico/treinamento) e dados de produção (simulados)
# Usamos a mesma divisão do notebook (80/20)
reference_df, current_df = train_test_split(df_ml, test_size=0.20, random_state=42)

print(f"Tamanho do conjunto de Referência (Treino): {reference_df.shape[0]} amostras")
print(f"Tamanho do conjunto de Produção (Simulado): {current_df.shape[0]} amostras")

# 3. Simular Data Drift (Desvio de Dados) artificial em produção para testar o monitoramento
# Vamos introduzir alterações em 'Age' e 'CreditScore' para simular um envelhecimento da carteira 
# e uma redução do score de crédito dos novos clientes.
np.random.seed(42)
current_df = current_df.copy()

# Aumentar a média de idade em ~5 anos
current_df['Age'] = current_df['Age'] + np.random.randint(3, 8, size=len(current_df))

# Reduzir o score de crédito médio em ~40 pontos
current_df['CreditScore'] = current_df['CreditScore'] - np.random.randint(20, 60, size=len(current_df))
current_df['CreditScore'] = current_df['CreditScore'].clip(350, 850) # manter no intervalo válido

print("\n[Simulação] Injetado Data Drift artificial em produção:")
print(f" - Idade média (Referência): {reference_df['Age'].mean():.2f} anos | Produção: {current_df['Age'].mean():.2f} anos")
print(f" - Score de Crédito médio (Referência): {reference_df['CreditScore'].mean():.2f} | Produção: {current_df['CreditScore'].mean():.2f}")

# 4. Configurar e executar o Relatório do Evidently AI
print("\nProcessando relatórios de desvio com o Evidently AI...")
report = Report(metrics=[
    DataDriftPreset() # Preset que calcula o drift estatístico para todas as colunas
])

report.run(reference_data=reference_df, current_data=current_df)

# 5. Salvar o relatório interativo como arquivo HTML
caminho_relatorio = "reports/relatorio_estabilidade.html"
os.makedirs(os.path.dirname(caminho_relatorio), exist_ok=True)
report.save_html(caminho_relatorio)

print(f"\n[Sucesso] Relatório interativo de estabilidade gerado com sucesso!")
print(f"Caminho do arquivo: {os.path.abspath(caminho_relatorio)}")
print("Abra o arquivo HTML no seu navegador para ver o dashboard interativo do Evidently AI.")
