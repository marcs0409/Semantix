"""
Script de Inferência Standalone — Predição de Churn de Clientes Bancários
Projeto EBAC & Semantix

Este script simula o consumo do modelo treinado (XGBoost) em um ambiente de produção.
Ele carrega o artefato 'modelo_churn_xgboost.pkl', realiza as verificações de integridade
do modelo e valida os dados de entrada antes de gerar a inferência.
"""

import os
import joblib
import pandas as pd
from threshold_classifier import ThresholdClassifier


def verificar_modelo(artefato: dict):
    """
    Verifica a integridade do artefato de modelo carregado.
    Lança ValueError ou TypeError se o dicionário ou o pipeline estiverem corrompidos.
    """
    if not isinstance(artefato, dict):
        raise TypeError("O artefato carregado deve ser um dicionário serializado.")
    
    chaves_obrigatorias = ['pipeline', 'limiar_otimizado', 'colunas_features']
    for chave in chaves_obrigatorias:
        if chave not in artefato:
            raise KeyError(f"Artefato inválido. A chave obrigatória '{chave}' não foi encontrada.")
            
    pipeline = artefato['pipeline']
    if not hasattr(pipeline, 'predict_proba'):
        raise AttributeError("O pipeline contido no artefato deve implementar o método 'predict_proba'.")
        
    return True


def carregar_modelo(caminho_modelo='modelo_churn_xgboost.pkl'):
    """Carrega e valida o artefato serializado do modelo e metadados."""
    if not os.path.exists(caminho_modelo):
        raise FileNotFoundError(f"Arquivo de modelo '{caminho_modelo}' não encontrado. Execute o notebook primeiro.")
    
    artefato = joblib.load(caminho_modelo)
    verificar_modelo(artefato)
    return artefato


def prever_churn_cliente(dados_cliente: pd.DataFrame, caminho_modelo='modelo_churn_xgboost.pkl'):
    """
    Realiza a predição de churn para um ou mais clientes.
    
    Valida as colunas obrigatórias e tipos de dados de entrada antes de executar o pipeline.
    Retorna um DataFrame contendo a probabilidade de churn e o status final
    baseado no limiar de decisão otimizado.
    """
    if not isinstance(dados_cliente, pd.DataFrame):
        raise TypeError("O parâmetro 'dados_cliente' deve ser uma instância de pandas DataFrame.")
        
    if dados_cliente.empty:
        raise ValueError("O DataFrame de entrada 'dados_cliente' está vazio.")

    artefato = carregar_modelo(caminho_modelo)
    pipeline = artefato['pipeline']
    limiar = artefato.get('limiar_otimizado', 0.5)
    colunas_esperadas = artefato.get('colunas_features', [])

    # Validação de colunas obrigatórias
    colunas_faltantes = [col for col in colunas_esperadas if col not in dados_cliente.columns]
    if colunas_faltantes:
        raise ValueError(f"Dados de entrada incompletos. As seguintes colunas obrigatórias estão ausentes: {colunas_faltantes}")

    # Garantir a ordem exata das colunas esperadas pelo modelo
    X_input = dados_cliente[colunas_esperadas].copy()

    # Validação de valores nulos
    if X_input.isnull().any().any():
        raise ValueError("Os dados de entrada contêm valores nulos/ausentes. Realize o tratamento antes da inferência.")

    # Gerar probabilidades da classe positiva (1 = Churn)
    probabilidades = pipeline.predict_proba(X_input)[:, 1]
    
    # Aplicar o limiar de decisão otimizado
    predicoes = (probabilidades >= limiar).astype(int)
    
    resultados = dados_cliente.copy()
    resultados['Probabilidade_Churn'] = (probabilidades * 100).round(2)
    resultados['Predicao_Churn'] = predicoes
    resultados['Status_Predito'] = resultados['Predicao_Churn'].map({0: 'Permanecer', 1: 'Churn (Risco)'})
    
    colunas_saida = [col for col in ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'Probabilidade_Churn', 'Status_Predito'] if col in resultados.columns]
    return resultados[colunas_saida]


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
