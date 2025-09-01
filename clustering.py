from typing import List
from cluster import Cluster
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import networkx as nx
class Clustering:
    def __init__(self, cluster: List[Cluster], k: int, cost: float, costAtMost: float, costAtLeast: float):
        self.cluster = cluster
        self.k = int(k)
        self.cost = float(cost)
        self.costAtMost = float(costAtMost)
        self.costAtLeast = float(costAtLeast)

    def __repr__(self):
        return (
            f"Clustering(k={self.k}, cost={self.cost}, "
            f"costAtMost={self.costAtMost}, costAtLeast={self.costAtLeast})"
        )
    @classmethod
    def clustering_hierarchique(cls, points):
        """Calcule et affiche le dendrogramme hiérarchique"""
        try:
            # Conversion des points en array numpy
            points_array = np.array([[p.x, p.y] for p in points])
            
            # Calcul du clustering
            Z = linkage(points_array, method='ward')
            
            # Création du plot
            plt.figure(figsize=(10, 5))
            dendrogram(Z)
            plt.title("Clustering Hiérarchique")
            plt.ylabel("Distance")
            plt.xlabel("Points index")
            plt.show()
            
        except Exception as e:
            raise Exception(f"Erreur lors du clustering: {str(e)}")
    
  