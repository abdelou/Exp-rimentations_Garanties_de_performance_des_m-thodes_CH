
from espace_metrique import Espace_metrique
from tkinter import Tk, Canvas, Frame, Text, Scrollbar,simpledialog, BOTH, RIGHT, LEFT,X, Y, WORD, LAST, N, E, END, TOP, Toplevel, GROOVE, SUNKEN, Button, DISABLED,simpledialog
import tkinter.messagebox as messagebox
import numpy as np
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
import traceback
import itertools 
from  espace_metrique  import Espace_metrique
from clustering_optimal import Clustering_Optimal
class Choix_random:
    def __init__(self, parent):
        self.parent = parent
        self.parent.title("Visualisation des arbres π et π'")
        self.parent.geometry("800x800")
        
        # Variables pour stocker les données des arbres
        self.points_array = None
        self.D = None
        self.selected_indices = None
        self.R_values = None
        self.parent_indices = None
        self.pi_prime = None
        self.edge_lengths_pi = None

        # Configuration de la fenêtre principale
        self.main_frame = Frame(parent)
        self.main_frame.pack(expand=True, fill=BOTH, padx=10, pady=10)

        # Boutons en haut
        self.btn_frame = Frame(self.main_frame)
        self.btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(self.btn_frame,
                 text="Afficher arbres",
                 command=self.afficher_arbres).pack(side=LEFT, padx=5)
        
        #ttk.Button(self.btn_frame,
                 #text="Facteur",
                 #command=self.calculer_facteur).pack(side=LEFT, padx=5)
        # Frame pour les arbres
        ttk.Button(self.btn_frame,
                    text="Quitter",
                    style='TButton',
                    command=self.quitter_application).pack(fill=X,)
    
    
        
        self.arbres_frame = Frame(self.main_frame)
        self.arbres_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def quitter_application(self):
        """Ferme proprement l'application"""
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter l'application ?"):
            self.parent.destroy()  # Ferme la fenêtre principale
            # Si vous avez d'autres processus à arrêter, ajoutez-les ici

    def afficher_arbres(self):
        """Affiche les arbres π et π' avec des points aléatoires"""
        # Générer des points aléatoires (entre 3 et 10 points)
        n_points = random.randint(3, 10)
        self.points_array = np.random.randint(2, 8, size=(n_points, 2))
        
        # Nettoyer l'affichage précédent
        if hasattr(self, 'canvas_arbres') and self.canvas_arbres:
            self.canvas_arbres.get_tk_widget().destroy()
        
        # Calculer les arbres
        while True:
            self.D = distance_matrix(self.points_array, self.points_array, p=2)
            
            self.selected_indices, self.R_values, self.parent_indices = Choix_random.farthest_first_tree(self.points_array)
            self.pi_prime, self.edge_lengths_pi, _, _ = Choix_random.hierarchical_tree(
                self.D, self.selected_indices, self.R_values)
            
            if Choix_random.arbres_sont_differents(self.parent_indices, self.pi_prime, self.selected_indices):
                break

        # Création de la figure
        fig = plt.Figure(figsize=(10, 4), dpi=100)
        ax1 = fig.add_subplot(121)  # Premier arbre
        ax2 = fig.add_subplot(122)  # Deuxième arbre

        # Calcul des limites avec marge dynamique
        x_min, x_max = min(self.points_array[:, 0]), max(self.points_array[:, 0])
        y_min, y_max = min(self.points_array[:, 1]), max(self.points_array[:, 1])
        x_margin = (x_max - x_min) * 0.3  # 30% de marge
        y_margin = (y_max - y_min) * 0.3


        # Dessiner le premier arbre (π)
        ax1.scatter(self.points_array[:,0], self.points_array[:,1], c='gray', s=50)
        for k in range(1, len(self.selected_indices)):
            child = self.selected_indices[k]
            parent = self.parent_indices[k]
            ax1.plot([self.points_array[child,0], self.points_array[parent,0]], 
                    [self.points_array[child,1], self.points_array[parent,1]], 'k--', linewidth=1)
            mid_x = (self.points_array[child,0] + self.points_array[parent,0]) / 2
            mid_y = (self.points_array[child,1] + self.points_array[parent,1]) / 2
            ax1.text(mid_x, mid_y, f"{self.R_values[k]:.2f}", fontsize=8, color='blue')
            ax1.scatter(self.points_array[self.selected_indices[k],0],
                      self.points_array[self.selected_indices[k],1], c='red', s=50)
        
        for k, idx in enumerate(self.selected_indices):
            ax1.text(self.points_array[idx,0], self.points_array[idx,1], f"{k+1}", fontsize=10, color='red')
        
        ax1.set_title("Arbre π (Farthest-First)")
        ax1.grid(True)
        ax1.set_xlim(x_min - x_margin, x_max + x_margin)  # Marge dynamique
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        ax1.set_aspect('equal')  # Ratio carré

        # Dessiner le deuxième arbre (π')
        ax2.scatter(self.points_array[:,0], self.points_array[:,1], c='gray', s=50)
        for (child_num, parent_num), length in self.edge_lengths_pi.items():
            child_idx = self.selected_indices[child_num-1]
            parent_idx = self.selected_indices[parent_num-1]
            ax2.plot([self.points_array[child_idx,0], self.points_array[parent_idx,0]], 
                    [self.points_array[child_idx,1], self.points_array[parent_idx,1]], 'k-', linewidth=1)
            mid_x = (self.points_array[child_idx,0] + self.points_array[parent_idx,0]) / 2
            mid_y = (self.points_array[child_idx,1] + self.points_array[parent_idx,1]) / 2
            ax2.text(mid_x, mid_y, f"{length:.2f}", fontsize=8, color='blue')
        
        for k, idx in enumerate(self.selected_indices):
            #ax2.scatter(self.points_array[idx,0]+0.5, self.points_array[idx,1], c='red', s=50)
            ax2.text(self.points_array[idx,0], self.points_array[idx,1], f"{k+1}", fontsize=10, color='red')
        
        ax2.set_title("Arbre π' (Hiérarchique)")
        ax2.grid(True)
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)  # Marge dynamique
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        ax2.set_aspect('equal')

        plt.tight_layout()
        plt.subplots_adjust(left=0.5, right=0.95, top=0.9, bottom=0.1)
        # Intégration dans Tkinter

        self.canvas_arbres = FigureCanvasTkAgg(fig, master=self.arbres_frame)
        self.canvas_arbres.draw()
        self.canvas_arbres.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def calculer_cout(self):
        """Calcule et affiche le coût de clustering"""
        if not hasattr(self, 'points_array') or self.points_array is None:
            messagebox.showerror("Erreur", "Veuillez d'abord générer des arbres")
            return
        
        # Demander le nombre de clusters
        nb_clusters = tk.simpledialog.askinteger("Nombre de clusters", 
                                            "Entrez le nombre de clusters (k):",
                                            parent=self.parent,
                                            minvalue=2, maxvalue=len(self.points_array)-1)
        if nb_clusters is None:  # L'utilisateur a annulé
            return

        # Calculer les coûts
        resultats_df = self.comparer_k_clustering_evolutif(
            self.points_array, 
            self.selected_indices, 
            self.parent_indices, 
            self.pi_prime, 
            self.edge_lengths_pi, 
            nb_clusters
        )

        # Afficher les résultats dans une nouvelle fenêtre
        self.afficher_resultats(resultats_df)

    def afficher_resultats(self, resultats_df):
        print("Début de afficher_resultats")  # Debug
        try:
            # Calcul du facteur
            cost_ff = resultats_df.loc[resultats_df["Méthode"] == "Farthest-First Traversal", "Coût final"].values[0]
            cost_pi = resultats_df.loc[resultats_df["Méthode"] == "Arbre hiérarchique T^π′", "Coût final"].values[0]
            facteur = cost_ff / cost_pi if cost_pi != 0 else float('nan')
            print(f"Facteur calculé: {facteur}")  # Debug

            # Création fenêtre
            result_window = tk.Toplevel(self.parent)
            result_window.title("Résultats du calcul de coût")
            result_window.geometry("800x600")
        
            # Frame principal
            main_frame = tk.Frame(result_window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
            # Affichage facteur
            facteur_frame = tk.Frame(main_frame)
            facteur_frame.pack(fill=tk.X, pady=5)
            tk.Label(facteur_frame, 
                text=f"Facteur (Coût FF / Coût π′): {facteur:.4f}",
                font=('Arial', 12, 'bold')).pack()
        
            # Text widget avec scrollbar
            text_frame = tk.Frame(main_frame)
            text_frame.pack(fill=tk.BOTH, expand=True)
        
            scrollbar = tk.Scrollbar(text_frame)
            text_area = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_area.pack(fill=tk.BOTH, expand=True)
        
            # Insertion données
            text_area.insert(tk.END, "=== DÉTAIL DES COÛTS ===\n\n")
            for _, row in resultats_df.iterrows():
                text_area.insert(tk.END, f"Méthode: {row['Méthode']}\n")
                text_area.insert(tk.END, f"Coût final: {row['Coût final']:.4f}\n")
                text_area.insert(tk.END, f"Points: {row['Points impliqués']}\n")
                text_area.insert(tk.END, f"Cluster: {row['Cluster']}\n")
                text_area.insert(tk.END, "-"*50 + "\n")
        
            text_area.config(state=tk.DISABLED)
            print("Résultats affichés avec succès")  # Debug
        
        except Exception as e:
            print(f"ERREUR dans afficher_resultats: {str(e)}")  # Debug
            messagebox.showerror("Erreur", f"Impossible d'afficher les résultats: {str(e)}")
    
    @staticmethod
    def farthest_first_tree(points):
        D = distance_matrix(points, points, p=2)
        n_points = len(points)
        remaining_indices = list(range(n_points))
        first_index = np.random.choice(remaining_indices)
        selected_indices = [first_index]
        remaining_indices.remove(first_index)
        R_values = [np.inf]
        parent_indices = [None]

        for step in range(1, n_points):
            dists_to_selected = D[:, selected_indices]
            min_dists = np.min(dists_to_selected, axis=1)
            min_dists[selected_indices] = -1
            next_index = np.argmax(min_dists)
            Ri = min_dists[next_index]
            R_values.append(Ri)
            parent_distances = D[next_index, selected_indices]
            parent_index = selected_indices[np.argmin(parent_distances)]
            parent_indices.append(parent_index)
            selected_indices.append(next_index)
            remaining_indices.remove(next_index)

        return  selected_indices, R_values, parent_indices

    @staticmethod
    def hierarchical_tree(D, selected_indices, R_values, alpha=1, beta=2):
        n_points = len(selected_indices)
        R = alpha * R_values[1]
        granularity_levels = {0: [1]}
        point_levels = [0]

        for i in range(1, n_points):
            Ri = R_values[i]
            j = 1
            while Ri <= R / (beta ** (j-1)):
                j += 1
                if j>1000:
                    break
            point_levels.append(j)

            if j not in granularity_levels:
                granularity_levels[j] = []
            granularity_levels[j].append(i+1)

        pi_prime = {}
        edge_lengths = {}
        for i in range(1, n_points):
            level_i = point_levels[i]
            candidates = []
            for k in range(0, level_i):
                candidates.extend(granularity_levels.get(k, []))
            candidate_indices = [selected_indices[c-1] for c in candidates]
            distances = D[selected_indices[i], candidate_indices]
            min_idx = np.argmin(distances)
            closest_num = candidates[min_idx]
            pi_prime[i+1] = closest_num
            edge_lengths[(i+1, closest_num)] = distances[min_idx]

        return pi_prime, edge_lengths, point_levels, granularity_levels
    @staticmethod
    def arbres_sont_differents(parent_indices, pi_prime, selected_indices):
        for i in range(1, len(parent_indices)):
            parent_in_ff = selected_indices.index(parent_indices[i]) + 1
            if pi_prime.get(i+1) != parent_in_ff:
                return True
        return False
    @staticmethod
    def comparer_k_clustering_evolutif(points, selected_indices, parent_indices, pi_prime, edge_lengths_pi, nb_clusters):
        n_points = len(points)

        # --- Construction du graphe Farthest-First Traversal ---
        G_ff = nx.Graph()
        for i in range(1, n_points):
            child = selected_indices[i]
            parent = parent_indices[i]
            num_child = selected_indices.index(child) + 1
            num_parent = selected_indices.index(parent) + 1
            weight = np.linalg.norm(points[child] - points[parent])
            G_ff.add_edge(num_child, num_parent, weight=weight)

        # --- Construction du graphe T^π′ ---
        G_pi = nx.Graph()
        for (child_num, parent_num), weight in edge_lengths_pi.items():
            G_pi.add_edge(child_num, parent_num, weight=weight)

        # --- Fonction de suppression progressive et calcul des coûts ---
        def clustering_cost_evolutif(G1, G2):
            edges_sorted_ff = sorted(G1.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            edges_sorted_pi = sorted(G2.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
            G1_copy, G2_copy = G1.copy(), G2.copy()

            # Tableau des coûts finaux
            results = []

            for i in range(nb_clusters - 1):
                # Suppression dans Farthest-First
                u_ff, v_ff, _ = edges_sorted_ff[i]
                G1_copy.remove_edge(u_ff, v_ff)

                # Suppression dans T^π′
                u_pi, v_pi, _ = edges_sorted_pi[i]
                G2_copy.remove_edge(u_pi, v_pi)

            # Fonction pour calculer le coût maximal du clustering actuel
            def calc_cost(G):
                comps = list(nx.connected_components(G))
                max_diameter = 0
                point_pair = (None, None)
                cluster_idx = None
                for idx, comp in enumerate(comps):
                    comp_list = list(comp)
                    if len(comp_list) >= 2:
                        comp_points = np.array([points[selected_indices[num-1]] for num in comp_list])
                        D_comp = distance_matrix(comp_points, comp_points)
                        diameter = np.max(D_comp)
                        if diameter > max_diameter:
                            max_diameter = diameter
                            i, j = np.unravel_index(np.argmax(D_comp), D_comp.shape)
                            point_pair = (comp_list[i], comp_list[j])
                            cluster_idx = idx + 1
                return max_diameter, point_pair, cluster_idx

            # Coût final Farthest-First
            cost_ff, pair_ff, cluster_ff = calc_cost(G1_copy)
            # Coût final T^π′
            cost_pi, pair_pi, cluster_pi = calc_cost(G2_copy)

            
            # Enregistrer dans le tableau des résultats
            results.append({
                'Méthode': 'Farthest-First Traversal',
                'Coût final': cost_ff,
                'Points impliqués': pair_ff,
                'Cluster': cluster_ff
            })
            results.append({
                'Méthode': "Arbre hiérarchique T^π′",
                'Coût final': cost_pi,
                'Points impliqués': pair_pi,
                'Cluster': cluster_pi
            })
            
            return results

        # Exécuter le clustering évolutif et récupérer les coûts
        resultats = clustering_cost_evolutif(G_ff, G_pi)
        resultats_df = pd.DataFrame(resultats)
        coords_ff = {node: tuple(points[selected_indices[node-1]]) for node in G_ff.nodes()}
        coords_pi = {}
        for node in G_pi.nodes():
            if 1 <= node <= len(selected_indices):
                coords_pi[node] = tuple(points[selected_indices[node-1]])

        # On retourne aussi les coordonnées pour réutilisation
        #return resultats_df, coords_ff, coords_pi
        resultats = clustering_cost_evolutif(G_ff, G_pi)
        resultats_df = pd.DataFrame(resultats)

        # --- Sauvegarder les coordonnées des sommets ---
        coords_ff = {node: tuple(points[selected_indices[node-1]]) for node in G_ff.nodes()}
        coords_pi = {}
        for node in G_pi.nodes():
            if 1 <= node <= len(selected_indices):
                coords_pi[node] = tuple(points[selected_indices[node-1]])

        # Retourner résultats + coordonnées
        return resultats_df, coords_ff, coords_pi
    @staticmethod
    def calculer_facteur_static(parent=None, nb_experiences=None):
        """Version statique qui peut être appelée sans instance"""
        if nb_experiences is None:
            nb_experiences = simpledialog.askinteger("Nombre d'expériences", 
                                                   "Entrez le nombre d'expériences:",
                                                   parent=parent,
                                                   minvalue=1, maxvalue=1000)
            if nb_experiences is None:
                return None

        resultats_globaux = []
        
        for iteration in range(1, nb_experiences + 1):
            # ... (le reste de votre code de calcul existant)
            
            # Retournez directement les résultats au lieu de les afficher
            resultats_globaux.append({
                "Itération": iteration,
                "Coût FF": cost_ff,
                "Coût π'": cost_pi,
                "cout_diametre":cout_diametre,
                "Ratio FF/π′": ratio
            })
        
        return pd.DataFrame(resultats_globaux)


    @staticmethod
    def calculer_facteur(parent, nb_experiences, nb_points, nb_clusters, axe_x, axe_y):
        """Calcule le facteur cout(pi)/cout(pi') pour une seule expérience"""
        try:
            while True:
                CoûtOptimal,points,clusters =Clustering_Optimal.meilleure_clustering_optimal(nb_points, nb_clusters, axe_x, axe_y)
                points_optimale=points
                cout_diametre,p1, p2 = Clustering_Optimal.cout_max_diametre(clusters, points)
                print("celui la le cout de diametre ",cout_diametre,p1, p2)
                #Clustering_Optimal.afficher_clusters(points,clusters)
                # Calculer les arbres
                if not isinstance(points, np.ndarray):
                    points = np.array(points, dtype=float)

                clusters = [list(map(int, cluster)) for cluster in clusters]

                if not clusters:
                    raise ValueError("Aucun cluster généré")
            
            # Vérification que tous les points sont numériques
                if not np.issubdtype(points.dtype, np.number):
                    raise ValueError("Les points doivent être numériques")
            
                try:

                    D = distance_matrix(points, points, p=2)
                except Exception as e:
                    print(f"Erreur dans distance_matrix - points: {points}")
                    print(f"Type points: {type(points)}, shape: {points.shape if hasattr(points, 'shape') else 'N/A'}")
                    raise e
                
                selected_indices, R_values, parent_indices = Choix_random.farthest_first_tree(points)
                pi_prime, edge_lengths_pi, _, _ = Choix_random.hierarchical_tree(
                    D, selected_indices, R_values)
            
                if Choix_random.arbres_sont_differents(parent_indices, pi_prime, selected_indices):
                    break

            # Calculer les coûts pour le nombre de clusters spécifié
            resultats_df, coords_ff, coords_pi = Choix_random.comparer_k_clustering_evolutif(
                points, selected_indices, parent_indices, 
                pi_prime, edge_lengths_pi, nb_clusters)
            #Choix_random.afficher_arbre_fft(parent, points, selected_indices, parent_indices, R_values)

        # Assignation des valeurs
            # Sauvegarde des résultats
          
            parent.coords_ff=coords_ff
            parent.coords_pi=coords_pi
            parent.clusters=clusters
            parent.points=points.tolist()
            parent.parent_indices=parent_indices
            parent.R_values=R_values
            parent.selected_indices=selected_indices
            parent.points_optimale=points_optimale
            parent.D=D
            parent.cout_diametre=cout_diametre
           # parent.facteur_cout_similaire=facteur_cout_similaire
            cost_ff = resultats_df.loc[
                resultats_df["Méthode"] == "Farthest-First Traversal", 
                "Coût final"].values[0]
            cost_pi = resultats_df.loc[
                resultats_df["Méthode"] == "Arbre hiérarchique T^π′", 
                "Coût final"].values[0]
            
            # Calculer le ratio
            facteur = cost_pi / CoûtOptimal if CoûtOptimal != 0 else np.nan
            facteur_cout_similaire=cost_pi / cout_diametre if cout_diametre != 0 else np.nan
            print(f"Sauvegarde réussie - Nombre de clusters: {len(clusters)}") 
            print("=== DataFrame final ===")
            print({
                    "Coût FF": cost_ff,
                    "Coût π": cost_pi,
                    "CoûtOptimal": CoûtOptimal,
                    "Coût diamètre": cout_diametre,
                    "Facteur π/Coût diamètre": facteur_cout_similaire,
                    "Facteur π/CoûtOptimal": facteur
                    })

            
            return pd.DataFrame({
                "Itération": [1],
                "Coût FF": [float(cost_ff)],
                "Coût π": [float(cost_pi)],
                "CoûtOptimal": [float(CoûtOptimal)],
                "cout_diametre": [float(cout_diametre)],
                "facteur: π/cout_diametre": [float(facteur_cout_similaire)if not np.isnan(facteur_cout_similaire) else np.nan],
                "facteur: π/CoûtOptimal": [float(facteur) if not np.isnan(facteur) else np.nan]

            })
        except Exception as e:
            print(f"Erreur dans calculer_facteur: {str(e)}")
            traceback.print_exc()
            return pd.DataFrame()  # Retourne un DataFrame vide en cas d'erreur
    
    def afficher_arbres_depuis_donnees(self, points_array, selected_indices, parent_indices, R_values, pi_prime, edge_lengths_pi):
    
        # Supprimer l'ancien canvas s'il existe
        if hasattr(self, 'canvas_arbres') and self.canvas_arbres:
            self.canvas_arbres.get_tk_widget().destroy()

    # Création de la figure avec deux sous-graphes côte à côte
        fig = plt.Figure(figsize=(10, 5), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    # Calcul dynamique des marges
        x_min, x_max = np.min(points_array[:, 0]), np.max(points_array[:, 0])
        y_min, y_max = np.min(points_array[:, 1]), np.max(points_array[:, 1])
        x_margin = (x_max - x_min) * 0.3
        y_margin = (y_max - y_min) * 0.3

    # Dessin arbre π (Farthest-First)
        ax1.scatter(points_array[:, 0], points_array[:, 1], c='gray', s=50)
        for k in range(1, len(selected_indices)):
            child = selected_indices[k]
            parent = parent_indices[k]
            ax1.plot([points_array[child, 0], points_array[parent, 0]], 
                 [points_array[child, 1], points_array[parent, 1]], 'k--', linewidth=1)
            mid_x = (points_array[child, 0] + points_array[parent, 0]) / 2
            mid_y = (points_array[child, 1] + points_array[parent, 1]) / 2
            ax1.text(mid_x, mid_y, f"{R_values[k]:.2f}", fontsize=8, color='blue')
            ax1.scatter(points_array[child, 0], points_array[child, 1], c='red', s=50)

        for k, idx in enumerate(selected_indices):
            ax1.text(points_array[idx, 0], points_array[idx, 1], f"{k+1}", fontsize=10, color='red')

        ax1.set_title("Arbre π (Farthest-First)")
        ax1.grid(True)
        ax1.set_xlim(x_min - x_margin, x_max + x_margin)
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        ax1.set_aspect('equal')

    # Dessin arbre π' (Hiérarchique)
        ax2.scatter(points_array[:, 0], points_array[:, 1], c='gray', s=50)
        for (child_num, parent_num), length in edge_lengths_pi.items():
            child_idx = selected_indices[child_num - 1]
            parent_idx = selected_indices[parent_num - 1]
            ax2.plot([points_array[child_idx, 0], points_array[parent_idx, 0]],
                    [points_array[child_idx, 1], points_array[parent_idx, 1]], 'k-', linewidth=1)
            mid_x = (points_array[child_idx, 0] + points_array[parent_idx, 0]) / 2
            mid_y = (points_array[child_idx, 1] + points_array[parent_idx, 1]) / 2
            ax2.text(mid_x, mid_y, f"{length:.2f}", fontsize=8, color='blue')

        for k, idx in enumerate(selected_indices):
            ax2.text(points_array[idx, 0], points_array[idx, 1], f"{k+1}", fontsize=10, color='red')

        ax2.set_title("Arbre π′ (Hiérarchique)")
        ax2.grid(True)
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        ax2.set_aspect('equal')

        plt.tight_layout()

    # Création du canvas et intégration dans la frame dédiée
        self.canvas_arbres = FigureCanvasTkAgg(fig, master=self.arbres_frame)
        self.canvas_arbres.draw()
        self.canvas_arbres.get_tk_widget().pack(fill=tk.BOTH, expand=True)


    def afficher_resultats_facteur(self, tableau_final):
        """Affiche les résultats du calcul de facteur"""
        result_window = Toplevel(self.parent)
        result_window.title("Résultats du facteur de coût")
        result_window.geometry("800x600")

        # Cadre principal
        main_frame = Frame(result_window)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        # Texte résumé
        summary_frame = Frame(main_frame)
        summary_frame.pack(fill=tk.X, pady=5)
    
        facteur_max = tableau_final['facteur π′/CoûtOptimal'].max()
        facteur_moyen = tableau_final['facteur π′/CoûtOptimal'].mean()
    
        tk.Label(summary_frame, 
            text=f"Ratio maximal: {facteur_max:.4f} | Ratio moyen: {facteur_moyen:.4f}",
            font=('Arial', 12, 'bold')).pack()

        # Tableau des résultats
        tree_frame = Frame(main_frame)
        tree_frame.pack(fill=BOTH, expand=True)

        # Création du Treeview
        tree = ttk.Treeview(tree_frame, columns=list(tableau_final.columns), show="headings")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Configuration des colonnes
        for col in tableau_final.columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')

        # Ajout des données
        for _, row in tableau_final.iterrows():
            tree.insert("", "end", values=list(row))

        # Placement des widgets
        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Configuration du redimensionnement
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)