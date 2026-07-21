import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.validation import check_is_fitted


class ThresholdClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper para classificadores do scikit-learn que otimiza o limiar de decisão
    para maximizar o F1-score usando validação cruzada out-of-fold durante o fit.
    """
    def __init__(self, base_estimator, cv=5, threshold_range=None, pos_label=1):
        self.base_estimator = base_estimator
        self.cv = cv
        self.threshold_range = threshold_range
        self.pos_label = pos_label

    def fit(self, X, y):
        # 1. Validar se o problema é estritamente binário
        classes_unicas = np.unique(y)
        if len(classes_unicas) != 2:
            raise ValueError(f"ThresholdClassifier suporta apenas problemas de classificação binária. Foram encontradas {len(classes_unicas)} classes.")

        self.classes_ = classes_unicas
        if self.pos_label not in self.classes_:
            raise ValueError(f"A classe positiva pos_label={self.pos_label} não foi encontrada nas classes de y: {self.classes_}")

        pos_idx = np.where(self.classes_ == self.pos_label)[0][0]

        # 2. Verificar suporte a predict_proba ou calibrar se possuir apenas decision_function
        estimator_para_cv = clone(self.base_estimator)
        
        has_proba = hasattr(estimator_para_cv, "predict_proba")
        has_decision = hasattr(estimator_para_cv, "decision_function")

        if not has_proba:
            if has_decision:
                # Usar CalibratedClassifierCV para calibrar decision_function em probabilidades válidas
                estimator_para_cv = CalibratedClassifierCV(
                    estimator=estimator_para_cv, method='sigmoid', cv=self.cv
                )
            else:
                raise AttributeError("O estimador base deve implementar 'predict_proba' ou 'decision_function'.")

        # 3. Obter probabilidades out-of-fold sem fit inicial redundante
        cv_inner = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        probs_val = cross_val_predict(
            estimator_para_cv, X, y, cv=cv_inner, method='predict_proba', n_jobs=None
        )[:, pos_idx]

        # 4. Otimizar o limiar sobre dados de treino/validação
        th_range = self.threshold_range if self.threshold_range is not None else np.linspace(0.1, 0.9, 81)

        melhor_limiar = 0.5
        melhor_f1 = -1.0

        for th in th_range:
            preds = np.where(probs_val >= th, self.pos_label, self.classes_[1 - pos_idx])
            score = f1_score(y, preds, pos_label=self.pos_label, zero_division=0)
            if score > melhor_f1:
                melhor_f1 = score
                melhor_limiar = th

        self.threshold_ = melhor_limiar
        self.pos_idx_ = pos_idx

        # 5. Ajustar o estimador final no conjunto completo (X, y)
        if not has_proba and has_decision:
            self.estimator_ = CalibratedClassifierCV(
                estimator=clone(self.base_estimator), method='sigmoid', cv=self.cv
            ).fit(X, y)
        else:
            self.estimator_ = clone(self.base_estimator).fit(X, y)

        return self

    def _get_fitted_estimator(self):
        check_is_fitted(self, attributes=['threshold_'])
        if hasattr(self, 'estimator_'):
            return self.estimator_
        if hasattr(self, 'base_estimator'):
            check_is_fitted(self.base_estimator)
            return self.base_estimator
        raise AttributeError("Nenhum estimador ajustado encontrado.")

    def predict_proba(self, X):
        est = self._get_fitted_estimator()
        return est.predict_proba(X)

    def predict(self, X):
        est = self._get_fitted_estimator()
        pos_idx = getattr(self, 'pos_idx_', 1)
        pos_label = getattr(self, 'pos_label', 1)
        classes = getattr(self, 'classes_', getattr(est, 'classes_', np.array([0, 1])))
        
        probs = est.predict_proba(X)[:, pos_idx]
        neg_label = classes[1 - pos_idx] if len(classes) == 2 else 0
        return np.where(probs >= self.threshold_, pos_label, neg_label)

    def __getattr__(self, name):
        # Evitar recursão em atributos privados/especiais e de serialização
        if name.startswith('_') or name in ('threshold_', 'classes_', 'estimator_', 'base_estimator', 'cv', 'threshold_range', 'pos_label', 'pos_idx_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Se o modelo já foi ajustado, delegar para estimator_ ou base_estimator
        if 'estimator_' in self.__dict__:
            return getattr(self.estimator_, name)
        if 'base_estimator' in self.__dict__:
            return getattr(self.base_estimator, name)
            
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
