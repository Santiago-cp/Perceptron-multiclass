import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Perceptrón Multiclase definido anteriormente
class PerceptronMulticlase:
    def __init__(self, tasa_aprendizaje=0.01, n_iteraciones=1000, n_clases=3):
        self.tasa_aprendizaje = tasa_aprendizaje
        self.n_iteraciones = n_iteraciones
        self.n_clases = n_clases
        self.funcion_activacion = self._funcion_escalon_unitario
        self.pesos = None
        self.sesgo = None

    def ajustar(self, X, y):
        n_muestras, n_caracteristicas = X.shape

        # Inicializar parámetros para cada clase
        self.pesos = np.zeros((self.n_clases, n_caracteristicas))
        self.sesgo = np.zeros(self.n_clases)

        for _ in range(self.n_iteraciones):
            for idx, x_i in enumerate(X):
                salida_lineal = np.dot(self.pesos, x_i) + self.sesgo
                y_predicho = self.funcion_activacion(salida_lineal)

                # Convertir y a codificación one-hot
                y_verdadero = np.zeros(self.n_clases)
                y_verdadero[y[idx]] = 1

                # Actualizar pesos y sesgo para cada clase
                for i in range(self.n_clases):
                    actualizacion = self.tasa_aprendizaje * (y_verdadero[i] - y_predicho[i])
                    self.pesos[i] += actualizacion * x_i
                    self.sesgo[i] += actualizacion

    def predecir(self, X):
        salida_lineal = np.dot(X, self.pesos.T) + self.sesgo
        return np.argmax(self.funcion_activacion(salida_lineal), axis=1)

    def _funcion_escalon_unitario(self, x):
        return np.where(x >= 0, 1, 0)

    def graficar_frontera_decision(self, X, y, titulo="Frontera de Decisión del Perceptrón"):
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', marker='o')

        # Definir el rango del gráfico
        x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
        y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = self.predecir(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
        plt.title(titulo)
        plt.xlabel('Característica 1')
        plt.ylabel('Característica 2')
        plt.show()

# Diseño de la aplicación en Streamlit
def main():
    st.title("Red Neuronal Perceptrón con Simulación de Datos")

    # Contenedores para el diseño en columnas
    contenedor_izquierdo, contenedor_derecho = st.columns(2)

    # Contenedor izquierdo: Generación de datos
    with contenedor_izquierdo:
        st.subheader("Generar Conjunto de Datos Simulado")

        n_muestras = st.number_input('Número de simulaciones', value=500, key='muestras_input')
        n_centros = st.number_input('Número de clases', value=3, key='centros_input')

        if st.button('Generar'):
            # Generar datos con make_blobs
            X, y = make_blobs(n_samples=n_muestras, centers=n_centros, n_features=2, random_state=42)

            # Almacenar los datos generados en el estado de la sesión
            st.session_state.X = X
            st.session_state.y = y

            # Graficar los datos
            st.write("Conjunto de Datos Simulado:")
            fig, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', marker='o')
            ax.set_xlabel('Característica 1')
            ax.set_ylabel('Característica 2')
            st.pyplot(fig)

    # Contenedor derecho: Ajuste del modelo y frontera de decisión
    with contenedor_derecho:
        st.subheader("Estimar Red Neuronal Perceptrón")

        if 'X' in st.session_state and 'y' in st.session_state:  # Asegurarse de que los datos están almacenados en el estado de la sesión
            n_iteraciones = st.number_input('Número de iteraciones', value=1000, key='iteraciones_input')
            tasa_aprendizaje = st.number_input('Tasa de aprendizaje', value=0.01, key='tasa_aprendizaje_input')
            n_clases = st.number_input('Número de clases', value=n_centros, key='clases_input')

            if st.button('Estimar Red Neuronal'):
                # Recuperar datos del estado de la sesión
                X = st.session_state.X
                y = st.session_state.y

                # Ajustar el modelo de Perceptrón
                perceptron = PerceptronMulticlase(tasa_aprendizaje=tasa_aprendizaje, n_iteraciones=n_iteraciones, n_clases=n_clases)
                perceptron.ajustar(X, y)

                # Graficar la frontera de decisión
                st.write("Frontera de Decisión:")
                fig, ax = plt.subplots()
                perceptron.graficar_frontera_decision(X, y)
                st.pyplot(fig)
        else:
            st.warning("¡Primero genere el conjunto de datos!")

# Ejecutar la aplicación
if __name__ == '__main__':
    main()
