# Importar bibliotecas
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Criar um DataFrame de exemplo
# Para este exemplo, vamos criar um DataFrame fictício com dados aleatórios
np.random.seed(42)
data_size = 100
X = np.random.rand(data_size, 4)  # 4 features
y = np.random.randint(0, 3, data_size)  # 3 classes

# Dividir o dataset em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Escalonar os dados para melhorar o desempenho do modelo
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar o modelo usando Árvore de Decisão (Decision Tree)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular a acurácia
accuracy = accuracy_score(y_test, y_pred)

# Exibir a acurácia
print(f"Acurácia do modelo: {accuracy:.2f}")
