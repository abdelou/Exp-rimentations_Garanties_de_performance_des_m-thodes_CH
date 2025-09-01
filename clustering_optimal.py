import numpy as np
from math import inf
from espace_metrique import Espace_metrique
from tkinter import Tk, Canvas, Frame, Text, Scrollbar,simpledialog, BOTH, RIGHT, LEFT,X, Y, WORD, LAST, N, E, END, TOP, Toplevel, GROOVE, SUNKEN, Button, DISABLED,simpledialog
import tkinter.messagebox as messagebox
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from tkinter.simpledialog import askinteger
import tkinter as tk
from tkinter import ttk
import random
import networkx as nx
import pandas as pd
import itertools 
class Clustering_Optimal:
    def __init__(self):
        pass
    @staticmethod
    def meilleure_clustering_optimal(nb_points,k_cluster,axe_x,axe_y):
        

        points = np.random.uniform(1, max(axe_x, axe_y), size=(nb_points, 2))
        points_optimal = {i+1: list(coord) for i, coord in enumerate(points)}

      
        def k_center_optimal(points_dict, k_cluster):
            # ici on réparation des données à partir du dictionnaire
            labels = list(points_dict.keys())
            coords = np.array([points_dict[i] for i in labels])
            label_to_index = {label: idx for idx, label in enumerate(labels)}
            dist_matrix = distance_matrix(coords, coords)

            def max_radius(clusters, centers):
                max_dist = 0
                farthest_points = {}
                for c_idx, cluster in enumerate(clusters):
                    center = centers[c_idx]
                    center_idx = label_to_index[center]
                    cluster_max_dist = 0
                    farthest_point = None
                    for p in cluster:
                        dist = dist_matrix[label_to_index[p]][center_idx]
                        if dist > cluster_max_dist:
                            cluster_max_dist = dist
                            farthest_point = p
                        max_dist = max(max_dist, dist)
                    farthest_points[c_idx] = (farthest_point, cluster_max_dist)
                return max_dist, farthest_points

            # ici pour générer tous les  partitions possible 
            def all_partitions_k(elements, k_cluster):
                def helper(parts, remaining):
                    if len(parts) > k_cluster:
                        return
                    if not remaining:
                        if len(parts) == k_cluster:
                            yield parts
                        return
                    for i in range(len(parts)):
                        yield from helper(parts[:i] + [parts[i] + [remaining[0]]] + parts[i+1:], remaining[1:])
                    yield from helper(parts + [[remaining[0]]], remaining[1:])
                return helper([], elements)
                

            # ici c'est l'algorithme principal pour calculer le cout_optimal_clustering
            meilleur_clusters = None
            meilleur_centers = None
            min_cout = float('inf')
            meilleur_farthest_points = None
                    
            for partition in all_partitions_k(labels, k_cluster):
                if any(len(c) == 0 for c in partition):
                    continue
                for possible_centers in itertools.product(*partition):
                    cout, farthest_points = max_radius(partition, possible_centers)
                    if cout < min_cout:
                        min_cout = cout
                        meilleur_clusters = partition
                        meilleur_centers = possible_centers
                        meilleur_farthest_points = farthest_points
            return meilleur_clusters, meilleur_centers, min_cout, meilleur_farthest_points
                
        clusters, centers, cout, farthest_points = k_center_optimal(points_optimal, k_cluster)
        CoûtOptimal=round(cout, 3)
        print(f"Clusters trouvés : {clusters}")  
        print("les corrdoner de meilleur clustering",points.tolist() )
        return CoûtOptimal,points.tolist(),clusters
    


    @staticmethod
    def afficher_clusters(points, clusters):
        couleurs = ['red', 'blue', 'green', 'orange', 'purple']
        for idx, cluster in enumerate(clusters):
            cluster_points = np.array([points[i-1] for i in cluster])
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        color=couleurs[idx % len(couleurs)],
                        label=f"Cluster {idx+1}")
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Clustering Optimal")
        plt.show()

    @staticmethod
    def afficher_clusters_cecle(points, clusters):

        couleurs = ['red', 'blue', 'green', 'orange', 'purple']

        plt.figure(figsize=(8, 6))

        for idx, cluster in enumerate(clusters):
            cluster_points = np.array([points[i-1] for i in cluster])
            
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                        color=couleurs[idx % len(couleurs)],
                        label=f"Cluster {idx+1}")

        # ici on calculer le centre du cluster
            center = cluster_points.mean(axis=0)
        # ici on  calculer le rayon c'ets a dire la distance maximale par rapport le centre 
            radius = np.max(np.linalg.norm(cluster_points - center, axis=1))

        
            circle = plt.Circle(center, radius, color=couleurs[idx % len(couleurs)],
                            fill=False, linestyle='--', linewidth=2, alpha=0.5)
            plt.gca().add_patch(circle)

        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Clustering Optimal avec cercles")
        plt.axis('equal')
        plt.show()
    @staticmethod
    def cout_max_diametre(clusters, points_dict):
        max_diametre = -1
        point1_max = None
        point2_max = None
        # vérification  si points_dict est une liste ou un dict
        if isinstance(points_dict, list):
          
            points_dict = {i + 1: pt for i, pt in enumerate(points_dict)}

        for cluster in clusters:
            if len(cluster) < 2:
                continue  
            try:
                coords = np.array([points_dict[pid] for pid in cluster])
            except KeyError as e:
                print(f"Identifiant de point introuvable : {e}")
                continue
            dist_mat = distance_matrix(coords, coords)
            diametre = np.max(dist_mat)
            

            i, j = np.unravel_index(np.argmax(dist_mat), dist_mat.shape)

            candidate_p1 = coords[i].tolist()
            candidate_p2 = coords[j].tolist()

            if diametre > max_diametre:
                max_diametre = diametre
                point1_max = candidate_p1
                point2_max = candidate_p2

        return round(max_diametre, 3),point1_max,point2_max
