# 💓 Previsão de Doença Cardíaca com Machine Learning

## 🚀 Projeto de Machine Learning para Diagnóstico Cardiovascular

Este projeto tem como objetivo analisar dados clínicos e aplicar modelos de Machine Learning para prever a presença de doenças cardíacas em pacientes, utilizando o famoso dataset `heart.csv`.

---

## 🔍 Passo a passo do projeto

1. **📥 Importação dos dados**  
   Importamos o dataset com informações clínicas relevantes, como idade, pressão arterial, colesterol, sexo e sintomas.

2. **📊 Análise Exploratória (EDA)**  
   - Distribuição da doença cardíaca por faixa etária  
   - Comparação dos níveis de colesterol entre homens e mulheres  
   - Comparativo da ocorrência da doença entre sexos  
   - Heatmap mostrando a correlação entre as variáveis

3. **⚙️ Pré-processamento dos dados**  
   - Transformação de variáveis categóricas em numéricas (One-Hot Encoding)  
   - Separação entre variáveis independentes (features) e alvo (HeartDisease)  
   - Divisão em dados de treino e teste (80/20)  
   - Padronização das features com StandardScaler para melhor desempenho dos modelos

4. **🤖 Treinamento e Otimização dos Modelos**  
   - K-Nearest Neighbors (KNN) com GridSearchCV para encontrar o melhor número de vizinhos  
   - Random Forest como modelo de ensemble para comparação

5. **📈 Avaliação dos modelos**  
   - Cálculo da acurácia em conjunto de teste  
   - Matrizes de confusão para análise detalhada  
   - Relatórios de classificação com precisão, recall e f1-score  
   - Gráficos comparativos mostrando desempenho dos modelos

---

## 💡 Resultados

- Melhor número de vizinhos no KNN encontrado via GridSearchCV  
- Avaliação comparativa: KNN vs Random Forest  
- Visualização clara da performance e interpretação dos resultados para o contexto clínico
