#Importando as Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

#Configurando para o terminal mostrar todas as colunas
pd.set_option('display.max_columns', None)

#Importando o Modelo
df = pd.read_csv('heart.csv')

#Verificando algumas informações
print(df.columns)
print(df.isnull().sum())
print(df.info())

#Convertendo as strings em valores numéricos
df_nums = pd.get_dummies(df, drop_first=True)
print(df_nums.columns)

#Gráficos para melhor visualização das informações
sns.histplot(data=df, x='Age', hue='HeartDisease')
plt.title('Doença no coração por idade')
plt.xlabel('Idade')
plt.ylabel('Total')
plt.show()

sns.boxplot(data=df, y='Cholesterol', hue='Sex')
plt.title('Média de Colesterol por Sexo')
plt.xlabel('Sexo')
plt.ylabel('Colesterol')
plt.show()

sns.countplot(data=df, x='Sex', hue='HeartDisease')
plt.title('Comparativo entre Masculino e Feminino')
plt.xlabel('')
plt.ylabel('Total')
plt.show()

sns.heatmap(data=df.select_dtypes(include='number').corr(), annot=True)
plt.title('Relação entre as variáveis')
plt.show()

#Separando Variáveis e Resultado
X = df_nums[['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak',
             'Sex_M', 'ChestPainType_ATA', 'ChestPainType_NAP',
             'ChestPainType_TA', 'RestingECG_Normal', 'RestingECG_ST',
             'ExerciseAngina_Y', 'ST_Slope_Flat', 'ST_Slope_Up']]
y = df_nums['HeartDisease']

#Separando o Dataset entre Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=9)

#Criando os Pipelines
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())
])

#Utilizando o GridSearch para o modelo de knn
knn_model = GridSearchCV(estimator=knn_pipeline,
                     param_grid={'knn__n_neighbors': range(1,10)},
                     cv=3)
knn_model.fit(X_train,y_train)

#Verificando os nomes das colunas do knn
knn_results = pd.DataFrame(knn_model.cv_results_)
print(knn_results.columns)

#Gráfico para visualização dos resultados de diferentes k em KNN
sns.pointplot(data=knn_results, x='param_knn__n_neighbors', y='mean_test_score')
plt.title('Precisão média do KNN por número de vizinhos')
plt.xlabel('Número de Vizinhos')
plt.ylabel('Precisão Média')
plt.show()

#Criando o modelo de Random Forest
rf_pipeline.fit(X_train,y_train)
rf_pred = rf_pipeline.predict(X_test)

#Separando o melhor resultado do knn para comparação
knn_pred = knn_model.best_estimator_.predict(X_test)

#Comparação entre os modelos
print('\nAvaliação dos Modelos:')

print(f'Precisão do KNN: {accuracy_score(y_test, knn_pred) * 100:.2f}%')
print(f'Precisão do Random Forest: {accuracy_score(y_test, rf_pred) * 100:.2f}%')

print('\nMatriz de Confusão do KNN:')
print(confusion_matrix(y_test, knn_pred))
print('\nMatriz de Confusão do RandomForest:')
print(confusion_matrix(y_test, rf_pred))

print('\nRelatório de Classificação do KNN:')
print(classification_report(y_test, knn_pred))
print('\nRelatório de Classificação do RandomForest:')
print(classification_report(y_test, rf_pred))

#Separando as informações para colocar no gráfico a seguir
model_names = ['KNN', 'Random Forest']
accuracies = [
    accuracy_score(y_test, knn_pred),
    accuracy_score(y_test, rf_pred)
]

#Gráficos para melhor visualização
ConfusionMatrixDisplay.from_predictions(y_test, knn_pred)
plt.title('Matriz de Confusão - KNN')
plt.show()

ConfusionMatrixDisplay.from_predictions(y_test, rf_pred)
plt.title('Matriz de Confusão - Random Forest')
plt.show()

sns.barplot(y=accuracies, hue=model_names)
plt.title('Comparação entre os modelos de ML')
plt.xlabel('')
plt.ylabel('Precisão')
plt.show()