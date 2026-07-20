# 📊 Prevenção e Predição de Churn de Clientes Bancários

**Projeto EBAC & Semantix**

---

## 📌 1. Visão Geral do Projeto & Problema de Negócio

No setor bancário e financeiro, a retenção de clientes é significativamente mais rentável do que a aquisição de novos clientes. O **Churn** (cancelamento/saída do cliente) representa uma perda direta de receita e valor de carteira.

Este projeto desenvolve uma solução completa de **Machine Learning de ponta a ponta** para prever a probabilidade de churn de clientes a partir de dados socioeconômicos e comportamentais (`Churn_Modelling.csv`).

### 🎯 Objetivos de Negócio:
1. **Identificar os principais fatores de risco** que levam o cliente ao encerramento da conta.
2. **Construir e comparar modelos preditivos avançados** (Dummy Classifier, Regressão Logística, Random Forest, XGBoost e SVM).
3. **Avaliar curvas de discriminação (ROC-AUC e PR-AUC)** para bases desbalanceadas.
4. **Otimizar o Limiar de Decisão (*Decision Threshold Tuning*)** para maximizar o **F1-Score** da classe de Churn.
5. **Exportar o modelo campeão** e fornecer um script de inferência standalone (`predict.py`) para consumo em produção.

---

## 💡 2. Principais Insights da Análise Exploratória (EDA)

* **Disparidade Geográfica (Alemanha):** A taxa de churn na **Alemanha é de ~32.4%**, praticamente o **dobro** em relação à França (~16.2%) e Espanha (~16.7%).
* **Efeito da Engajamento (Membros Ativos):** Clientes inativos apresentam quase **o dobro da taxa de churn** (26.9%) em comparação a membros ativos (14.2%).
* **Ponto Crítico de Produtos:** Clientes com **3 produtos atingem mais de 80% de churn**, e com **4 produtos a taxa é de 100%**. Isso indica uma forte insatisfação ou sobrecarga de ofertas/tarifas.
* **Score de Crédito Extremo:** Scores de crédito inferiores a 400 possuem taxa de churn de 100%.
* **Idade:** Clientes na faixa entre **35 e 60 anos** concentram a maior proporção de cancelamentos.

---

## 🛠️ 3. Metodologia de Machine Learning & MLOps

A esteira de modelagem seguiu os mais rigorosos padrões da indústria para evitar vazamento de dados (*Data Leakage*):

1. **Divisão Estratificada (Train/Test Split):** 80% treino e 20% teste com preservação da proporção da classe minoritária (`stratify=y`).
2. **Engenharia de Features:** 
   - `IsBalanceZero`: Indicador binário para saldo zerado.
   - `Age_x_IsActiveMember`: Interação entre idade e nível de atividade do membro.
3. **Pré-processamento com `ColumnTransformer` & `Pipeline`:** 
   - Padronização de variáveis numéricas (`StandardScaler`).
   - One-Hot Encoding para variáveis categóricas (`OneHotEncoder`).
4. **Baseline Naïve (`DummyClassifier`):** Comparação científica provando o valor gerado pelos modelos sobre um palpite ingênuo.
5. **Otimização via `GridSearchCV`:** Busca exaustiva de hiperparâmetros combinada com validação cruzada estratificada em 5 dobras (`StratifiedKFold`).
6. **Ajuste Fino de Limiar (*Decision Threshold Optimization*):** Busca de limiar ideal via predições *Out-of-Fold* (`cross_val_predict`) para maximização direta da métrica **F1-Score**.
7. **Avaliação por Curvas ROC-AUC e PR-AUC:** Análise gráfica da capacidade discriminatória dos modelos em bases desbalanceadas.
8. **Interpretabilidade com SHAP:** Uso de valores SHAP (*TreeExplainer*) para explicar a contribuição global e individual de cada feature no modelo.
9. **Inferência Standalone (`predict.py`):** Script Python pronto para produção simulando o consumo do modelo por microsserviços ou APIs.

---

## 📈 4. Comparativo de Desempenho dos Modelos (Dados de Teste)

| Modelo | Acurácia | Precisão (Churn) | Recall (Churn) | **F1-Score (Churn)** | Limiar Otimizado |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline Naïve)** | 79.60% | 0.00% | 0.00% | 0.00% | 0.50 |
| **Regressão Logística (Baseline)** | 71.90% | 39.28% | **69.78%** | 50.27% | 0.50 |
| **Random Forest (Otimizado)** | 84.50% | 62.09% | 61.18% | 61.63% | 0.55 |
| **SVM (Otimizado)** | **84.85%** | **62.68%** | 63.14% | 62.91% | 0.38 |
| **🏆 XGBoost (Otimizado - Campeão)** | 84.70% | 62.00% | 64.13% | **63.04%** | **0.58** |

### 🥇 Justificativa do Modelo Campeão:
Embora o *Dummy Classifier* apresente 79.60% de acurácia (por prever apenas a classe majoritária), ele possui **F1-Score de 0.00%**, provando que acurácia isolada é insuficiente em bases desbalanceadas. 
O **XGBoost Otimizado** apresentou a melhor combinação entre precisão (62.00%) e recall (64.13%), atingindo o maior **F1-Score absoluto (63.04%)**.

---

## 📁 5. Estrutura do Repositório

```text
├── Churn_Modelling.csv          # Base de dados original
├── projeto semantix churn.ipynb # Notebook Jupyter com a esteira completa
├── modelo_churn_xgboost.pkl     # Artefato do modelo campeão exportado (joblib)
├── predict.py                   # Script de inferência standalone para produção
├── requirements.txt             # Dependências do projeto
└── README.md                    # Documentação executiva do projeto
```

---

## 🚀 6. Como Executar o Projeto

1. **Clonar o Repositório:**
   ```bash
   git clone <URL_DO_REPOSITORIO>
   cd Semantix
   ```

2. **Instalar as Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Executar o Notebook:**
   Abra o Jupyter Notebook ou PyCharm e execute as células sequencialmente em `projeto semantix churn.ipynb`.

4. **Executar Teste de Inferência via Linha de Comando:**
   ```bash
   python predict.py
   ```

---
*Projeto desenvolvido em Data Science na EBAC em parceria com a Semantix.*