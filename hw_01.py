import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import imageio


iris = load_iris()
X = iris.data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Найдем оптимальное количество кластеров с помощью метода локтя
inertia = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Метод локтя для выбора оптимального количества кластеров')
plt.xlabel('Количество кластеров')
plt.ylabel('Сумма')
plt.show()

optimal_k = 4

# Реализуем алгоритм kmeans с визуализацией каждого шага
def kmeans_custom(X, k, max_iters=100, tol=1e-4):

    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    filenames = []
    for i in range(max_iters):
        # Присваиваем точки к ближайшему центроиду
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Вычисляем новые центроиды
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(k)])

        # Визуализация на каждом шаге
        plt.figure(figsize=(8, 6))
        for j in range(k):
            plt.scatter(X[labels == j][:, 0], X[labels == j][:, 1], label=f'Cluster {j + 1}')
        plt.scatter(new_centroids[:, 0], new_centroids[:, 1], c='black', marker='x', label='Centroids')
        plt.title(f'Iteration {i + 1}')
        plt.legend()

        # Сохраняем изображение на каждом шаге
        filename = f"iteration_{i + 1}.png"
        filenames.append(filename)
        plt.savefig(filename)
        plt.close()

        # Проверка на сходимость
        if np.all(np.abs(new_centroids - centroids) < tol):
            print(f"Converged after {i + 1} iterations")
            break
        centroids = new_centroids
    return centroids, labels, filenames



centroids, labels, filenames = kmeans_custom(X_scaled, optimal_k)

with imageio.get_writer('kmeans_steps.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
import os
for filename in filenames:
    os.remove(filename)
print("GIF создан и сохранен как 'kmeans_steps.gif'")
