from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import f1_score
import numpy as np

class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper para classificadores do scikit-learn que otimiza o limiar de decisão
    para maximizar o F1-score usando validação cruzada out-of-fold durante o fit.
    """
    def __init__(self, base_estimator, cv=5, threshold_range=None):
        self.base_estimator = base_estimator
        self.cv = cv
        self.threshold_range = threshold_range
        
    def fit(self, X, y):
        # 1. Ajustar o classificador base no conjunto completo de treino fornecido a esta dobra
        self.base_estimator.fit(X, y)
        
        # 2. Obter probabilidades fora-da-dobra (out-of-fold) para otimização do limiar
        cv_inner = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        try:
            # Usando n_jobs=None (execução sequencial interna) para evitar concorrência com o laço do GridSearchCV
            probs_val = cross_val_predict(
                self.base_estimator, X, y, cv=cv_inner, method='predict_proba', n_jobs=None
            )[:, 1]
        except (AttributeError, ValueError):
            # Fallback caso o estimador base não implemente predict_proba ou ocorra erro
            if hasattr(self.base_estimator, "decision_function"):
                dec_func = self.base_estimator.decision_function(X)
                probs_val = (dec_func - dec_func.min()) / (dec_func.max() - dec_func.min() + 1e-9)
            else:
                raise AttributeError("O estimador base deve implementar predict_proba ou decision_function.")
                
        th_range = self.threshold_range if self.threshold_range is not None else np.linspace(0.1, 0.9, 81)
        
        melhor_limiar = 0.5
        melhor_f1 = 0.0
        
        for th in th_range:
            preds = (probs_val >= th).astype(int)
            score = f1_score(y, preds)
            if score > melhor_f1:
                melhor_f1 = score
                melhor_limiar = th
                
        self.threshold_ = melhor_limiar
        self.classes_ = self.base_estimator.classes_
        return self
        
    def predict_proba(self, X):
        return self.base_estimator.predict_proba(X)
        
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= self.threshold_).astype(int)

    def __getattr__(self, name):
        # Delegar atributos não encontrados (como feature_importances_, coef_ etc.) ao estimador base
        if name != 'base_estimator':
            return getattr(self.base_estimator, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
