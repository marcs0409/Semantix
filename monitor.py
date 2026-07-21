import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Suporte resiliente a diferentes versões do Evidently AI
try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
except ImportError:
    try:
        from evidently.legacy.report import Report
        from evidently.legacy.metric_preset import DataDriftPreset
    except ImportError:
        raise ImportError("O pacote 'evidently' não foi encontrado. Instale-o com 'pip install evidently'.")


def main():
    parser = argparse.ArgumentParser(description="Rotina de Monitoramento de Data Drift com Evidently AI")
    parser.add_argument(
        "--simular-drift",
        action="store_true",
        help="Se ativado, injeta um desvio artificial (envelhecimento da carteira e queda no score) nos dados de produção."
    )
    args = parser.parse_args()

    print("=== INICIANDO ROTINA DE MONITORAMENTO (EVIDENTLY AI) ===")

    caminho_csv = "data/Churn_Modelling.csv"
    if not os.path.exists(caminho_csv):
        raise FileNotFoundError(f"Arquivo '{caminho_csv}' não encontrado. Certifique-se de que a pasta /data está correta.")

    df = pd.read_csv(caminho_csv, delimiter=',')

    # 1. Aplicar exatamente a mesma esteira de Feature Engineering do treinamento
    df_ml = df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], errors='ignore')
    df_ml['Gender'] = df_ml['Gender'].map({'Female': 0, 'Male': 1})
    df_ml['IsBalanceZero'] = (df_ml['Balance'] == 0).astype(int)
    df_ml['Age_x_IsActiveMember'] = df_ml['Age'] * df_ml['IsActiveMember']

    # Garantir exatamente o mesmo schema de 12 variáveis preditoras do modelo
    colunas_features = [
        'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
        'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
        'EstimatedSalary', 'IsBalanceZero', 'Age_x_IsActiveMember'
    ]
    df_features = df_ml[colunas_features].copy()

    # 2. Dividir em dados de referência (treino) e dados de produção (simulados)
    reference_df, current_df = train_test_split(df_features, test_size=0.20, random_state=42)

    print(f"Tamanho do conjunto de Referência (Treino): {reference_df.shape[0]} amostras")
    print(f"Tamanho do conjunto de Produção (Teste):     {current_df.shape[0]} amostras")

    # 3. Injeção de Data Drift artificial somente se a flag for ativada
    if args.simular_drift:
        print("\n[Simulação de Drift] Flag --simular-drift ativada. Injetando desvio artificial...")
        np.random.seed(42)
        current_df = current_df.copy()
        current_df['Age'] = current_df['Age'] + np.random.randint(3, 8, size=len(current_df))
        current_df['CreditScore'] = (current_df['CreditScore'] - np.random.randint(20, 60, size=len(current_df))).clip(350, 850)
        # Recalcular a feature derivada com a nova idade
        current_df['Age_x_IsActiveMember'] = current_df['Age'] * current_df['IsActiveMember']

        print(f" - Idade média (Referência): {reference_df['Age'].mean():.2f} anos | Produção: {current_df['Age'].mean():.2f} anos")
        print(f" - Score de Crédito médio (Referência): {reference_df['CreditScore'].mean():.2f} | Produção: {current_df['CreditScore'].mean():.2f}")
    else:
        print("\n[Monitoramento Real] Processando dados sem alteração artificial.")

    # 4. Configurar e executar o Relatório do Evidently AI
    print("\nProcessando relatórios de estabilidade estatística com o Evidently AI...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    # 5. Salvar o relatório interativo como arquivo HTML
    caminho_relatorio = "reports/relatorio_estabilidade.html"
    os.makedirs(os.path.dirname(caminho_relatorio), exist_ok=True)
    report.save_html(caminho_relatorio)

    print(f"\n[Sucesso] Relatório interativo de estabilidade gerado com sucesso!")
    print(f"Caminho do arquivo: {os.path.abspath(caminho_relatorio)}")
    print("Abra o arquivo HTML no seu navegador para ver o dashboard interativo do Evidently AI.")


if __name__ == '__main__':
    main()
