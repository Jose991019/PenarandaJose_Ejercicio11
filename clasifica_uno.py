import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn

numeros = skdata.load_digits()
target = numeros['target']
imagenes = numeros['images']
n_imagenes = len(target)


data = imagenes.reshape((n_imagenes, -1)) 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
x_train, x_test, y_train, y_test = train_test_split(data, target, train_size=0.5)
y_train[y_train!=1] = 0
y_test[y_test!=1]=0

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

numero = 1
dd = y_train==numero
cov = np.cov(x_train[dd].T)
valores, vectores = np.linalg.eig(cov)
valores = np.real(valores)
vectores = np.real(vectores)
ii = np.argsort(-valores)
valores = valores[ii]
vectores = vectores[:,ii]

clf = LinearDiscriminantAnalysis()

predicciones_train = []
predicciones_test = []
proyeccion_train = np.dot(x_train,vectores)
proyeccion_test = np.dot(x_test,vectores)
F1Scores_train = []
F1Scores_test = []
F1_segunda_grafica_train = []
F1_segunda_grafica_test = []
for i in range(3,41):
    clf.fit(proyeccion_train[:,:i], y_train.T)
    predicciones_train = clf.predict(proyeccion_train[:,:i])
    predicciones_test = clf.predict(proyeccion_test[:,:i])
    F1Scores_train.append(sklearn.metrics.f1_score(y_train,predicciones_train))
    F1Scores_test.append(sklearn.metrics.f1_score(y_test,predicciones_test))
    F1_segunda_grafica_train.append(sklearn.metrics.f1_score(y_train,predicciones_train, pos_label = 0))
    F1_segunda_grafica_test.append(sklearn.metrics.f1_score(y_test,predicciones_test, pos_label = 0))
    
x = range(3,41)

plt.figure(figsize = (10,5))
plt.subplot(1,2,1)
plt.scatter(x,F1Scores_train,label = 'Train')
plt.scatter(x,F1Scores_test, label = 'Test')
plt.legend()
plt.title('Clasificacion UNO')
plt.xlabel('Número de componentes PCA')
plt.ylabel('F1 Score')
plt.subplot(1,2,2)
plt.scatter(x,F1_segunda_grafica_train,label = 'Train')
plt.scatter(x,F1_segunda_grafica_test, label = 'Test')
plt.legend()
plt.title('Clasificacion otros')
plt.xlabel('Número de componentes PCA')
plt.ylabel('F1 Score')
plt.savefig('F1_score_LinearDiscriminantAnalysis.png')