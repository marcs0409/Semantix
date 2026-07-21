import unittest
import pandas as pd
import os
import joblib
from predict import carregar_modelo, prever_churn_cliente
from threshold_classifier import ThresholdClassifier  # Necessário para carregar o modelo sem erro de classe

class TestPredict(unittest.TestCase):
    def setUp(self):
        self.caminho_modelo = 'modelo_churn_xgboost.pkl'
        
        # Cliente simulado com formato correto esperado pelo pipeline
        self.cliente_valido = pd.DataFrame([{
            'CreditScore': 600,
            'Geography': 'France',
            'Gender': 1,
            'Age': 40,
            'Tenure': 3,
            'Balance': 60000.0,
            'NumOfProducts': 2,
            'HasCrCard': 1,
            'IsActiveMember': 1,
            'EstimatedSalary': 50000.0,
            'IsBalanceZero': 0,
            'Age_x_IsActiveMember': 40
        }])
        
    def test_carregar_modelo(self):
        """Valida se o modelo existe e é carregado com as chaves corretas."""
        if os.path.exists(self.caminho_modelo):
            artefato = carregar_modelo(self.caminho_modelo)
            self.assertIsInstance(artefato, dict)
            self.assertIn('pipeline', artefato)
            self.assertIn('limiar_otimizado', artefato)
            self.assertIn('colunas_features', artefato)
        else:
            self.skipTest(f"Modelo '{self.caminho_modelo}' não encontrado localmente.")
            
    def test_prever_churn_cliente(self):
        """Valida se a inferência retorna o formato de DataFrame esperado e colunas corretas."""
        if os.path.exists(self.caminho_modelo):
            resultado = prever_churn_cliente(self.cliente_valido, self.caminho_modelo)
            
            # Verificar tipo
            self.assertIsInstance(resultado, pd.DataFrame)
            
            # Verificar se as colunas essenciais de saída estão presentes
            colunas_esperadas = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'Probabilidade_Churn', 'Status_Predito']
            for col in colunas_esperadas:
                self.assertIn(col, resultado.columns)
                
            # Verificar probabilidade no intervalo válido
            prob = resultado['Probabilidade_Churn'].iloc[0]
            self.assertTrue(0 <= prob <= 100)
            
            # Verificar status
            status = resultado['Status_Predito'].iloc[0]
            self.assertIn(status, ['Permanecer', 'Churn (Risco)'])
        else:
            self.skipTest(f"Modelo '{self.caminho_modelo}' não encontrado localmente.")

if __name__ == '__main__':
    unittest.main()
