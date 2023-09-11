# Primeiro fazer as importações das bibliotecas

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import numpy as np
import plotly.express as px


# Vamos criar duas variáveis (as informações do x = idades, y = salário)
x = [21, 23, 27, 32, 34, 41, 50, 53, 57]
y = [1000, 1100, 1250, 1700, 2000, 2400, 2700, 3000, 3100]

# Agora vamos começar com as informações que desejamos
grafico = px.scatter(x=x, y=y)
grafico.show() # Aqui vamos visualizar o gráfico

# Variável referente ao salário
base_salario = np.array([[21,100], [23,1100], [27,1250], [32,1700], [34,2000], [41,2400], [50, 2700], [53, 3000], [57,3100]]) # as informações do array são X + Y (idades + salário)
base_salario # Visualizar

scaler_salario = StandardScaler() # Aqui estamos criando uma variável para passar para StandardScaler()
base_salario = scaler_salario.fit_transform(base_salario) # Aqui terá as informações concretas do array

kmeans_salario = KMeans(n_clusters=3)
kmeans_salario.fit(base_salario)

centro = kmeans_salario.cluster_centers_
centro # Visualizar

scaler_salario.inverse_transform(kmeans_salario.cluster_centers_) # Aqui invertemos 

rotulos = kmeans_salario.labels_

# Vamos criar as informações do gráfico agora
grafico1 = px.scatter(x = base_salario[:,0], y = base_salario[:,1}, color=rotulos)
grafico2 = px.scatter(x = centro[:,0], y = centro[:,1], size=[8,8,8])
grafico3 = go.Figure(data = grafico1.data + grafico2.data)
grafico3.show()


