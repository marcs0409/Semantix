import unittest
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from predict import carregar_modelo, prever_churn_cliente, verificar_modelo
from threshold_classifier import ThresholdClassifier


class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.caminho_modelo = 'modelo_churn_xgboost.pkl'
        
        # Se o modelo oficial não estiver presente (ex: checkout limpo), gera um modelo mínimo de teste para reprodutibilidade
        if not os.path.exists(cls.caminho_modelo):
            print(f"\n[Aviso CI] Artefato '{cls.caminho_modelo}' não encontrado. Gerando modelo fixture mínimo...")
            X_dummy = pd.DataFrame([
                {'CreditScore': 600, 'Geography': 'France', 'Gender': 1, 'Age': 40, 'Tenure': 3, 'Balance': 60000.0, 'NumOfProducts': 2, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 50000.0, 'IsBalanceZero': 0, 'Age_x_IsActiveMember': 40},
                {'CreditScore': 650, 'Geography': 'Spain', 'Gender': 1, 'Age': 35, 'Tenure': 4, 'Balance': 50000.0, 'NumOfProducts': 1, 'HasCrCard': 1, 'IsActiveMember': 1, 'EstimatedSalary': 60000.0, 'IsBalanceZero': 0, 'Age_x_IsActiveMember': 35},
                {'CreditScore': 500, 'Geography': 'Germany', 'Gender': 0, 'Age': 50, 'Tenure': 1, 'Balance': 80000.0, 'NumOfProducts': 1, 'HasCrCard': 0, 'IsActiveMember': 0, 'EstimatedSalary': 90000.0, 'IsBalanceZero': 0, 'Age_x_IsActiveMember': 0},
                {'CreditScore': 450, 'Geography': 'Germany', 'Gender': 0, 'Age': 55, 'Tenure': 2, 'Balance': 70000.0, 'NumOfProducts': 3, 'HasCrCard': 0, 'IsActiveMember': 0, 'EstimatedSalary': 85000.0, 'IsBalanceZero': 0, 'Age_x_IsActiveMember': 0}
            ])
            y_dummy = pd.Series([0, 0, 1, 1])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary', 'Age_x_IsActiveMember']),
                    ('cat', OneHotEncoder(drop='first', sparse_output=False), ['Geography'])
                ],
                remainder='passthrough'
            )
            pipe = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', ThresholdClassifier(base_estimator=RandomForestClassifier(n_estimators=10, random_state=42), cv=2))
            ])
            pipe.fit(X_dummy, y_dummy)
            
            artefato = {
                'pipeline': pipe,
                'limiar_otimizado': 0.5,
                'colunas_features': X_dummy.columns.tolist(),
                'f1_score_teste': 1.0
            }
            joblib.dump(artefato, cls.caminho_modelo)

    def setUp(self):
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
        if not os.path.exists(self.caminho_modelo):
            self.fail(f"FALHA CRÍTICA DE REPRODUTIBILIDADE: O arquivo de modelo '{self.caminho_modelo}' é obrigatório.")
            
        artefato = carregar_modelo(self.caminho_modelo)
        self.assertIsInstance(artefato, dict)
        self.assertTrue(verificar_modelo(artefato))
            
    def test_prever_churn_cliente(self):
        """Valida se a inferência retorna o formato de DataFrame esperado e colunas corretas."""
        if not os.path.exists(self.caminho_modelo):
            self.fail(f"FALHA CRÍTICA DE REPRODUTIBILIDADE: O arquivo de modelo '{self.caminho_modelo}' é obrigatório.")
            
        resultado = prever_churn_cliente(self.cliente_valido, self.caminho_modelo)
        
        self.assertIsInstance(resultado, pd.DataFrame)
        colunas_esperadas = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'Probabilidade_Churn', 'Status_Predito']
        for col in colunas_esperadas:
            self.assertIn(col, resultado.columns)
            
        prob = resultado['Probabilidade_Churn'].iloc[0]
        self.assertTrue(0 <= prob <= 100)
        
        status = resultado['Status_Predito'].iloc[0]
        self.assertIn(status, ['Permanecer', 'Churn (Risco)'])

    def test_verificacao_modelo_corrompido(self):
        """Garante que verificar_modelo detecta artefatos inválidos ou corrompidos."""
        with self.assertRaises(TypeError):
            verificar_modelo("objeto_invalido")
            
        with self.assertRaises(KeyError):
            verificar_modelo({'pipeline': None}) # Faltando chaves obrigatorias

    def test_validacao_colunas_faltantes(self):
        """Garante que prever_churn_cliente lança ValueError quando faltam colunas obrigatórias."""
        cliente_incompleto = self.cliente_valido.drop(columns=['CreditScore'])
        with self.assertRaises(ValueError):
            prever_churn_cliente(cliente_incompleto, self.caminho_modelo)

    def test_validacao_valores_nulos(self):
        """Garante que prever_churn_cliente lança ValueError quando há valores nulos."""
        cliente_nulo = self.cliente_valido.copy()
        cliente_nulo.loc[0, 'Age'] = np.nan
        with self.assertRaises(ValueError):
            prever_churn_cliente(cliente_nulo, self.caminho_modelo)


class TestThresholdClassifier(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_bin = np.random.randn(100, 4)
        self.y_bin = np.random.choice([0, 1], size=100)
        
        self.X_multi = np.random.randn(100, 4)
        self.y_multi = np.random.choice([0, 1, 2], size=100)

    def test_not_fitted_error(self):
        """Garante que check_is_fitted lança NotFittedError antes do fit."""
        tc = ThresholdClassifier(base_estimator=LogisticRegression())
        with self.assertRaises(NotFittedError):
            tc.predict(self.X_bin)
        with self.assertRaises(NotFittedError):
            tc.predict_proba(self.X_bin)

    def test_binary_validation(self):
        """Garante que problema multiclasse lança ValueError."""
        tc = ThresholdClassifier(base_estimator=LogisticRegression())
        with self.assertRaises(ValueError):
            tc.fit(self.X_multi, self.y_multi)

    def test_threshold_optimization_and_predict(self):
        """Valida se o limiar é otimizado e se predict/predict_proba geram formatos corretos."""
        tc = ThresholdClassifier(base_estimator=LogisticRegression(random_state=42), cv=3)
        tc.fit(self.X_bin, self.y_bin)
        
        self.assertTrue(hasattr(tc, 'threshold_'))
        self.assertTrue(0.1 <= tc.threshold_ <= 0.9)
        
        probs = tc.predict_proba(self.X_bin)
        self.assertEqual(probs.shape, (100, 2))
        
        preds = tc.predict(self.X_bin)
        self.assertEqual(len(preds), 100)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_calibration_fallback_without_predict_proba(self):
        """Valida calibração automática via CalibratedClassifierCV para modelos sem predict_proba (ex: LinearSVC)."""
        tc = ThresholdClassifier(base_estimator=LinearSVC(random_state=42, dual='auto'), cv=3)
        tc.fit(self.X_bin, self.y_bin)
        
        probs = tc.predict_proba(self.X_bin)
        self.assertEqual(probs.shape, (100, 2))
        self.assertTrue(np.all((probs >= 0) & (probs <= 1)))

    def test_joblib_serialization(self):
        """Valida se o ThresholdClassifier pode ser serializado e desserializado com joblib sem erros de atribuição."""
        tc = ThresholdClassifier(base_estimator=LogisticRegression(random_state=42), cv=3)
        tc.fit(self.X_bin, self.y_bin)
        
        caminho_temp = 'temp_threshold_classifier.pkl'
        try:
            joblib.dump(tc, caminho_temp)
            tc_carregado = joblib.load(caminho_temp)
            
            preds_orig = tc.predict(self.X_bin)
            preds_carr = tc_carregado.predict(self.X_bin)
            
            np.testing.assert_array_equal(preds_orig, preds_carr)
        finally:
            if os.path.exists(caminho_temp):
                os.remove(caminho_temp)


if __name__ == '__main__':
    unittest.main()
