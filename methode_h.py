import numpy as np
import math

def distance_euclidienne(point1, point2):
    return math.sqrt(sum((x - y)**2 for x, y in zip(point1, point2)))

def calculer_centroide(points, donnees):
    return np.mean([donnees[i] for i in points], axis=0)

def fusion_ward(cluster1, cluster2, donnees):
    n1, n2 = len(cluster1), len(cluster2)
    c1 = calculer_centroide(cluster1, donnees)
    c2 = calculer_centroide(cluster2, donnees)
    return (n1*n2)/(n1+n2) * distance_euclidienne(c1, c2)

def classification_ward(donnees):
    n = len(donnees)
    clusters = [[i] for i in range(n)]
    historique = []
    step = 1

    while len(clusters) > 1:
        min_dist = float('inf')
        best_pair = (0, 1)
        
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = fusion_ward(clusters[i], clusters[j], donnees)
                if dist < min_dist:
                    min_dist = dist
                    best_pair = (i, j)

        i, j = best_pair
        nouveau_cluster = clusters[i] + clusters[j]
        
        historique.append({
            'step': step,
            'distance': min_dist,
            'fusion': (clusters[i], clusters[j]),  # Garde les indices originaux
            'nouveau_cluster': nouveau_cluster
        })
        
        clusters[i] = nouveau_cluster
        del clusters[j]
        step += 1
    
    return historique