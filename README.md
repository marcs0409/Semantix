# 📊 Prevenção e Predição de Churn de Clientes Bancários

**Projeto de Ciência de Dados — EBAC & Semantix**

---

## 📌 Visão Geral do Projeto

No setor bancário e financeiro, a retenção de clientes é significativamente mais rentável do que a aquisição de novos clientes. O **Churn** (cancelamento ou encerramento da conta) representa uma perda direta de receita, redução de portfólio e aumento nos custos de captação.

Este projeto desenvolve uma solução completa e pronta para produção de **Machine Learning de ponta a ponta** para prever a probabilidade de churn de clientes bancários a partir de dados socioeconômicos e comportamentais, combinando rigor estatístico, interpretabilidade do modelo e otimização de custos operacionais do negócio.

---

## 📥 1. Coleta de Dados & Proveniência

Os dados utilizados neste projeto provêm de um repositório público de dados bancários de varejo:
* **Fonte Pública Oficial**: [Kaggle - Bank Customer Churn Dataset](https://www.kaggle.com/datasets/radheshyamkollipara/bank-customer-churn)
* **Licença**: **CC0: Public Domain** (Domínio Público / Dados Abertos)
* **Data de Acesso**: Julho de 2026
* **Arquivo Local**: [`data/Churn_Modelling.csv`](data/Churn_Modelling.csv)
* **Volume**: 10.000 registros (clientes individuais) e 14 atributos operacionais/demográficos.

### 🎯 Definição do Evento Positivo (*Target*)
A variável dependente alvo é **`Exited`**, onde definimos estritamente o **evento positivo** como:
* **`Exited = 1` (Evento Positivo / Churn)**: O cliente encerrou a conta / cancelou os serviços do banco (1.963 registros — **19.63%** da base).
* **`Exited = 0` (Evento Negativo / Retido)**: O cliente permanece ativo com conta aberta no banco (8.037 registros — **80.37%** da base).

### 📋 Dicionário de Atributos:

| Atributo | Tipo de Dado | Descrição / Significado no Negócio |
| :--- | :--- | :--- |
| `RowNumber` | Numérico | Número sequencial da linha no banco de dados. |
| `CustomerId` | Numérico | Identificador único numérico do cliente. |
| `Surname` | Texto | Sobrenome do cliente. |
| `CreditScore` | Numérico | Pontuação de crédito do cliente (350 a 850). |
| `Geography` | Categórico | País de residência do cliente (França, Espanha ou Alemanha). |
| `Gender` | Categórico | Gênero do cliente (Feminino / Masculino). |
| `Age` | Numérico | Idade do cliente em anos. |
| `Tenure` | Numérico | Anos de relacionamento contínuo do cliente com o banco (0 a 10 anos). |
| `Balance` | Numérico | Saldo bancário atual depositado na conta do cliente. |
| `NumOfProducts` | Numérico | Quantidade de produtos bancários contratados pelo cliente (1 a 4). |
| `HasCrCard` | Binário | Indicador se possui cartão de crédito ativo (1 = Sim, 0 = Não). |
| `IsActiveMember` | Binário | Indicador de membro ativo nos canais do banco (1 = Ativo, 0 = Inativo). |
| `EstimatedSalary` | Numérico | Estimativa de receita/salário anual do cliente. |
| **`Exited`** | Binário (Target) | **Evento Positivo (1 = Churn / Cancelou, 0 = Retido / Permanece).** |

---

## 🧹 2. Tratamento e Engenharia de Atributos

Para garantir a integridade estatística e evitar a contaminação do modelo com **vazamento de dados (*Data Leakage*)**, detalhamos a seguir todas as etapas de limpeza e transformação:

### 2.1 Tratamento de Valores Ausentes, Duplicados e Outliers
* **Valores Ausentes (Missing Values)**: Verificado via `df.isnull().sum()`. O conjunto de dados possui **0 valores nulos** em todas as 14 colunas.
* **Registros Duplicados**: Verificado via `df.duplicated().sum()`. A base possui **0 registros duplicados**.
* **Tratamento de Outliers**: A análise exploratória (Boxplots de `EstimatedSalary`, `Balance`, `Age` e `CreditScore`) identificou valores altos em contas bancárias e clientes idosos (ex: idades até 92 anos ou saldos expressivos). Como esses valores são **fenômenos legítimos do negócio bancário** (clientes de alta renda ou idosos), **nenhum registro foi excluído artificialmente**. O impacto de amplitudes discrepantes foi mitigado pela padronização (`StandardScaler`) e pela escolha de algoritmos robustos baseados em árvores de decisão (`XGBoost` e `RandomForest`).

### 2.2 Descarte de Colunas e Justificativas Técnicas
As seguintes colunas foram **excluídas da esteira de Machine Learning**:
1. `RowNumber`: Índice sequencial sem qualquer valor informacional ou preditivo.
2. `CustomerId`: Chave primária que não possui relação comportamental com a decisão de churn.
3. `Surname`: Variável textual de altíssima cardinalidade que causaria sobreajuste (*overfitting*) e introduziria vieses não éticos.
4. `CreditScore_Bins`, `Tenure_Bins`, `EstimatedSalary_Bins`: Faixas categóricas temporárias criadas **exclusivamente na fase de EDA** para visualização gráfica. Foram descartadas na modelagem para evitar redundância com as variáveis numéricas contínuas originais.

### 2.3 Codificação de Variáveis Categóricas
* **`Gender` (Mapeamento Binário Direto)**:
  Convertido para valores numéricos inteiros:
  Gender: 0 se Female (Feminino), 1 se Male (Masculino)
* **`Geography` (One-Hot Encoding)**:
  Aplicado o `OneHotEncoder(drop='first')`. A categoria `'France'` foi definida como baseline de referência, gerando duas variáveis binárias:
  * `Geography_Germany`: `1` se Alemanha, `0` caso contrário.
  * `Geography_Spain`: `1` se Espanha, `0` caso contrário.

### 2.4 Engenharia de Features (*Feature Engineering*)
Criamos duas variáveis sintéticas fundamentadas em hipóteses do negócio:
1. **`IsBalanceZero`**: Indicador binário que sinaliza se o cliente possui saldo exatamente igual a zero.
   IsBalanceZero = (Balance == 0).astype(int)
   * *Hipótese*: Clientes com saldo zero possuem uma dinâmica de fidelidade e custo de saída muito inferior àqueles com capital mantido no banco.
2. **`Age_x_IsActiveMember`**: Atributo de interação multiplicativa entre a idade e a atividade do cliente.
   Age_x_IsActiveMember = Age x IsActiveMember
   * *Hipótese*: A inatividade do membro (`IsActiveMember = 0`) é um fator de risco mais severo em faixas etárias mais elevadas.

### 2.5 Esteira de Pré-processamento e Divisão Estratificada
* **Divisão Treino/Teste (80/20)**: Realizada com `train_test_split(..., test_size=0.2, stratify=y, random_state=42)` garantindo a manutenção da proporção exata de ~19.63% de churn em ambos os conjuntos.
* **Garantia contra Data Leakage**: Utilizamos o `ColumnTransformer` onde o `StandardScaler` (para colunas numéricas) e o `OneHotEncoder` (para categóricas) foram **ajustados (`fit`) estritamente nos dados de treino**, sendo apenas aplicados (`transform`) aos dados de teste.

---

## 🤖 3. Modelagem e Hiperparametrização

Avaliamos cinco classificadores para determinar a melhor capacidade de separação do evento positivo (`Exited = 1`):

1. **Dummy Classifier (Baseline Naïve)**: Classificador ingênuo que sempre prevê a classe majoritária (`0`), servindo como piso de avaliação.
2. **Regressão Logística**: Modelo linear probabilístico baseline com ponderação de classes (`class_weight='balanced'`).
3. **Random Forest Classifier**: Ensemble de árvores de decisão.
4. **Support Vector Machine (SVC)**: Classificador de margem com kernel de base radial (RBF).
5. **XGBoost Classifier (Campeão)**: Algoritmo de Gradient Boosting com regularização avançada.

### ⚙️ Otimização via GridSearchCV e Decision Threshold Tuning
* **Busca por Hiperparâmetros**: Executada com `GridSearchCV` e validação cruzada estratificada em 5 dobras (`StratifiedKFold(n_splits=5)`).
* **Otimização do Limiar de Decisão (*Decision Threshold Optimization*)**:
  Por padrão, modelos de classificação utilizam um limiar de probabilidade fixo de `0.50`. Como a base é desbalanceada (~20% churn), implementamos a classe customizada [`threshold_classifier.py`](threshold_classifier.py) que extrai probabilidades fora-da-dobra (*out-of-fold*) via `cross_val_predict`, selecionando o limiar de decisão que maximiza diretamente a métrica **F1-Score**.

---

## 📈 4. Avaliação de Desempenho e Matriz de Custos

### 4.1 Tabela Oficial Consolidadora de Resultados Estatísticos

A tabela abaixo sintetiza o desempenho comparativo de todos os modelos avaliados sob sementes fixas (`random_state=42`), englobando amostras de treino (8.000) e teste (2.000), limiares otimizados, métricas globais e validação cruzada estratificada em 5 dobras (*5-fold CV*):

| Modelo | Amostras (Treino / Teste) | Limiar Utilizado | Acurácia | Precisão (Churn) | Recall (Churn) | **F1-Score (Churn)** | ROC-AUC | PR-AUC (AP) | Validação Cruzada (F1 Média ± DP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Classifier (Baseline Naïve)** | 8.000 / 2.000 | 0.50 | 79.65% | 0.00% | 0.00% | 0.00% | 0.5000 | 0.2035 | 0.0000 ± 0.0000 |
| **Regressão Logística (Otimizada)** | 8.000 / 2.000 | 0.57 | 78.15% | 47.00% | 57.74% | 51.82% | 0.7835 | 0.5433 | 0.4988 ± 0.0169 |
| **Random Forest (Otimizado)** | 8.000 / 2.000 | 0.49 | 84.05% | 60.58% | 61.92% | 61.24% | 0.8563 | 0.6846 | 0.6170 ± 0.0267 |
| **SVM (Otimizado)** | 8.000 / 2.000 | 0.35 | 84.05% | 60.05% | 64.62% | 62.25% | 0.8582 | 0.6757 | 0.6090 ± 0.0195 |
| **🏆 XGBoost (Otimizado - Campeão)** | 8.000 / 2.000 | **0.53** | 83.55% | 58.23% | **67.81%** | **62.66%** | **0.8662** | **0.7142** | **0.6312 ± 0.0193** |

> **Justificativa do Campeão**: Embora o Dummy Classifier apresente 79.65% de acurácia (por apenas prever que ninguém cancela), seu F1-Score é 0.00%. O **XGBoost Otimizado** atingiu a maior métrica de discriminação em bases desbalanceadas (**PR-AUC = 0.7142**, **ROC-AUC = 0.8662** e **F1-Score na Validação Cruzada = 0.6312 ± 0.0193**), capturando **67.81%** de todos os clientes em churn real no conjunto de teste.

### 4.2 Interpretabilidade do Modelo com SHAP
Utilizamos o **SHAP (SHapley Additive exPlanations)** via `TreeExplainer` para explicar a contribuição global de cada atributo no resultado:

![Gráfico de Resumo SHAP](images/shap_summary_plot.png)

* **Impactos Principais**:
  - **Idade (`Age`)**: É o principal fator isolado. Quanto maior a idade, maior o impacto positivo na probabilidade de churn.
  - **Quantidade de Produtos (`NumOfProducts`)**: Contratar 3 ou 4 produtos eleva dramaticamente o risco de cancelamento.
  - **Localização na Alemanha (`Geography_Germany`)**: Estar localizado na Alemanha aumenta significativamente o score de risco.
  - **Atividade (`IsActiveMember`)**: Ser membro ativo reduz fortemente o risco de churn.

### 4.3 Otimização da Matriz de Custos do Negócio & Simulação da Campanha

Em aplicações bancárias reais, os erros de classificação possuem custos financeiros altamente assimétricos. Avaliar apenas o F1-Score (que trata o erro estatístico de forma simétrica) pode não ser suficiente para maximizar o retorno do negócio.

#### 💵 Premissas Financeiras de Mercado:
* **Falso Positivo (FP) — Custo de 50,00**: Alarme falso (custo de contato/campanha preventiva de retenção gasto com cliente que não iria sair).
* **Falso Negativo (FN) — Custo de 500,00**: Perda definitiva da receita do cliente (LTV / Lifetime Value estimado no banco).
* **Verdadeiro Positivo (TP) — Valor Líquido Esperado de -200,00**: **Valor financeiro esperado** calculado sob a premissa de uma **taxa de sucesso assumida de 50%** no resgate do cliente (Valor Esperado = Custo de Ativação 50 - 50% x LTV Salvo 500 = -200,00). Trata-se de uma **expectativa teórica de retorno modelada sob premissas de mercado, e não de uma economia comprovada em produção**, a qual deve ser validada empiricamente por testes A/B na operação real.
* **Verdadeiro Negativo (TN) — Custo de 0,00**: Cliente retido sem custos adicionais.

---

### 📊 Consolidação dos Resultados Financeiros e Simulação da Campanha (2.000 Clientes no Teste)

A tabela abaixo mantém **estritamente separados** o Limiar Otimizado por F1-Score, o Limiar Otimizado por Custo, o Custo Esperado, a Economia Simulada e o Resultado Real/Simulado de uma Campanha de Retenção:

| Cenário de Decisão | Limiar | Clientes Contatados (FP + TP) | Clientes Salvos (50% TP) | Custo da Campanha | LTV Recuperado | **Custo Financeiro Esperado** | **Economia Simulada vs. Sem Modelo** | **Resultado Líquido Real da Campanha** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sem Modelo de Retenção (Passivo)** | 1.00 | 0 | 0 | 0,00 | 0,00 | 203.500,00 | 0,00 (Referência) | -203.500,00 |
| **Limiar Otimizado por F1-Score (Estatístico)** | **0.53** | 474 | 138 | 23.700,00 | 69.000,00 | 20.200,00 | 183.300,00 | -20.200,00 |
| **Limiar Padrão (Baseline 0.50)** | 0.50 | 519 | 144 | 25.950,00 | 72.250,00 | 12.700,00 | 190.800,00 | -12.700,00 |
| **🏆 Limiar Otimizado por Custo (Financeiro)** | **0.20** | **1.143** | **190** | **57.150,00** | **95.250,00** | **-25.100,00** (Ganho Líquido) | **228.600,00** | **+25.100,00** (Lucro Líquido) |

---

### 🔍 Discriminação Detalhada dos 5 Elementos Requeridos:

1. **Limiar Otimizado por F1-Score (`0.53`)**: Ponto ótimo do ponto de vista estatístico puro (validação cruzada), maximizando a média harmônica entre precisão (58.23%) e recall (67.81%).
2. **Limiar Otimizado por Custo (`0.20`)**: Ponto ótimo do ponto de vista econômico bancário. Como perder um cliente (FN = 500) é 10 vezes mais caro do que um alarme falso (FP = 50), o limiar financeiramente ideal é reduzido para **0.20**, aumentando a sensibilidade para capturar mais potenciais churners.
3. **Custo Financeiro Esperado**: Custo final acumulado calculado pela fórmula de prejuízo operacional (TN * 0) + (FP * 50) + (FN * 500) + (TP * -200). Sob o limiar F1-Score (0.53), o custo é de **20.200,00**; sob o limiar financeiro (0.20), o custo é negativo (**-25.100,00**), indicando retorno positivo sobre a perda inicial.
4. **Economia Simulada (vs. Sem Modelo)**: Economia de capital conquistada em relação à inércia do banco sem modelo (onde 407 clientes dariam churn sem intervenção = 203.500,00 de prejuízo). A economia simulada é de **183.300,00** no limiar F1-Score e atinge **228.600,00** no limiar financeiro.
5. **Resultado Real / Simulado da Campanha de Retenção**: Na prática da operação de retenção ativando 1.143 clientes elegíveis (probabilidade >= 0.20), o banco gasta 57.150,00 na campanha, recupera 95.250,00 em LTV salvo dos clientes convertidos (190 clientes), cobre os prejuízos dos FNs residuais e encerra o lote de testes com um **lucro líquido real de +25.100,00**.

![Otimização de Custos de Negócio](images/otimizacao_custo_financeiro.png)

---

## 💡 5. Conclusões e Recomendações Estratégicas

### 🎯 5.1 Recomendações Práticas para o Banco

1. **Plano de Retenção Específico para a Alemanha**: A taxa de churn na Alemanha é de **32.4%** (o dobro em relação à França e Espanha). Recomenda-se realizar uma auditoria de satisfação e competitividade tarifária focada na operação alemã.
2. **Campanha de Reativação de Membros Inativos**: Membros inativos apresentam **26.9% de churn** vs 14.2% em ativos. Criar réguas automáticas de engajamento no aplicativo e incentivos de relacionamento.
3. **Reestruturação de Pacotes Multi-Produto**: Clientes com 3 produtos possuem **mais de 80% de churn**, e com 4 produtos a taxa atinge **100%**. Recomenda-se simplificar a contratação de produtos cruzados para evitar sobrecarga de tarifas.
4. **Monitoramento da Faixa Etária de 35 a 60 anos**: Faixa etária com maior concentração absoluta de cancelamentos. Recomenda-se oferecer produtos de investimento e consultoria patrimonial de longo prazo.

### ⚙️ 5.2 MLOps & Governança em Produção

* **Monitoramento de Data Drift**: O script [`monitor.py`](monitor.py) utiliza o **Evidently AI** para monitorar eventuais desvios de distribuição na produção em relação à base de treino, salvando o dashboard interativo em [`reports/relatorio_estabilidade.html`](reports/relatorio_estabilidade.html).
* **Integração Contínua (CI/CD)**: Workflow automatizado em [`.github/workflows/python-tests.yml`](.github/workflows/python-tests.yml) executa a suíte de testes unitários a cada alteração no repositório.
* **Inferência Standalone**: O script [`predict.py`](predict.py) consome o modelo serializado [`modelo_churn_xgboost.pkl`](modelo_churn_xgboost.pkl) para realizar predições em produção.

---

## 📁 6. Estrutura do Repositório

```text
├── .github/
│   └── workflows/
│       └── python-tests.yml         # Esteira de CI via GitHub Actions
├── .gitignore                       # Arquivos ignorados no controle de versão
├── data/
│   └── Churn_Modelling.csv          # Base de dados do Kaggle
├── images/
│   ├── shap_summary_plot.png        # Gráfico de interpretabilidade SHAP
│   └── otimizacao_custo_financeiro.png # Gráfico de custo operacional vs limiar
├── reports/
│   └── relatorio_estabilidade.html  # Dashboard de monitoramento de Data Drift
├── threshold_classifier.py          # Wrapper customizado para otimização de limiar
├── projeto_semantix_churn.ipynb    # Notebook Jupyter com a esteira completa
├── modelo_churn_xgboost.pkl         # Artefato do modelo campeão exportado (joblib)
├── predict.py                       # Script de inferência standalone para produção
├── test_predict.py                  # Suíte de testes unitários para a inferência
├── monitor.py                       # Rotina de monitoramento de drift (Evidently AI)
├── run_notebook.py                  # Script para execução automatizada do notebook
├── requirements.txt                 # Dependências do projeto
└── README.md                        # Documentação executiva e metodológica
```

---

## 🚀 7. Como Executar o Projeto

1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com/marcs0409/Semantix
   cd Semantix
   ```

2. **Instalar as Dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Executar a Inferência de Teste:**
   ```bash
   python predict.py
   ```

4. **Executar a Suíte de Testes Unitários:**
   ```bash
   python -m unittest test_predict.py
   ```

5. **Gerar Relatório de Monitoramento de Data Drift:**
   ```bash
   python monitor.py
   ```

---
*Projeto desenvolvido no curso de Data Science da EBAC em parceria com a Semantix.*
