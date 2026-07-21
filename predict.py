"""
Script de Inferência Standalone — Predição de Churn de Clientes Bancários
Projeto EBAC & Semantix

Este script simula o consumo do modelo treinado (XGBoost) em um ambiente de produção.
Ele carrega o artefato 'modelo_churn_xgboost.pkl' e realiza a predição para novos clientes.
"""

import os
import joblib
import pandas as pd
from threshold_classifier import ThresholdClassifier


def carregar_modelo(caminho_modelo='modelo_churn_xgboost.pkl'):
    """Carrega o artefato serializado do modelo e metadados."""
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Arquivo de modelo '{caminho_modelo}' não encontrado. Execute o notebook primeiro.")
    
    artefato = joblib.load(caminho_modelo)
    return artefato


def prever_churn_cliente(dados_cliente: pd.DataFrame, caminho_modelo='modelo_churn_xgboost.pkl'):
    """
    Realiza a predição de churn para um ou mais clientes.
    
    Retorna um DataFrame contendo a probabilidade de churn e o status final
    baseado no limiar de decisão otimizado.
    """
    artefato = carregar_modelo(caminho_modelo)
    pipeline = artefato['pipeline']
    limiar = artefato.get('limiar_otimizado', 0.5)
    
    # Gerar probabilidades da classe positiva (1 = Churn)
    probabilidades = pipeline.predict_proba(dados_cliente)[:, 1]
    
    # Aplicar o limiar de decisão otimizado
    predicoes = (probabilidades >= limiar).astype(int)
    
    resultados = dados_cliente.copy()
    resultados['Probabilidade_Churn'] = (probabilidades * 100).round(2)
    resultados['Predicao_Churn'] = predicoes
    resultados['Status_Predito'] = resultados['Predicao_Churn'].map({0: 'Permanecer', 1: 'Churn (Risco)'})
    
    return resultados[['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'Probabilidade_Churn', 'Status_Predito']]


if __name__ == '__main__':
    print("=== TESTE DE INFERÊNCIA DE CHURN (SEMANTIX) ===")
    
    # Exemplo de cliente fictício para teste de produção
    cliente_simulado = pd.DataFrame([{
        'CreditScore': 619,
        'Geography': 'Germany',       # Alemanha (região com maior taxa de churn)
        'Gender': 0,                  # Female (0)
        'Age': 42,
        'Tenure': 2,
        'Balance': 83807.86,
        'NumOfProducts': 1,
        'HasCrCard': 1,
        'IsActiveMember': 0,          # Membro inativo
        'EstimatedSalary': 112542.58,
        'IsBalanceZero': 0,
        'Age_x_IsActiveMember': 0
    }])
    
    # Executar a inferência
    resultado = prever_churn_cliente(cliente_simulado)
    
    print("\n[Resultado da Inferência]:")
    print(resultado.to_string(index=False))
