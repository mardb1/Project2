import numpy as np
import random

def initialiser_centroides(donnees, k):
    indices = random.sample(range(len(donnees)), k)
    return np.array([donnees[i] for i in indices]), indices

def calculer_clusters(donnees, centroides):
    clusters = [[] for _ in range(len(centroides))]
    for i, point in enumerate(donnees):
        distances = [np.linalg.norm(point - c) for c in centroides]
        cluster_id = np.argmin(distances)
        clusters[cluster_id].append(i)
    return clusters

def update_centroides(donnees, clusters):
    nouveaux_centroides = []
    for cluster in clusters:
        nouveaux_centroides.append(np.mean([donnees[i] for i in cluster], axis=0))
    return np.array(nouveaux_centroides)

def kmeans(donnees, k=3, max_iterations=100):
    # Initialisation
    centroides, init_indices = initialiser_centroides(donnees, k)
    
    for _ in range(max_iterations):
        clusters = calculer_clusters(donnees, centroides)
        nouveaux_centroides = update_centroides(donnees, clusters)
        
        if np.allclose(centroides, nouveaux_centroides):
            break
            
        centroides = nouveaux_centroides
    
    # Formatage des résultats
    labels = np.zeros(len(donnees))
    for cluster_id, cluster in enumerate(clusters):
        for i in cluster:
            labels[i] = cluster_id
            
    return {
        'labels': labels,
        'centroides': centroides,
        'clusters': clusters,
        'init_indices': init_indices
    }