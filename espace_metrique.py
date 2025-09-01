from tkinter import Tk, Canvas, Frame, Text, Scrollbar, BOTH, RIGHT, LEFT, Y, WORD, LAST, N, E, END, TOP, Toplevel,GROOVE, SUNKEN,Button
import tkinter.messagebox as messagebox
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from point import Point
from clustering import Clustering
from tkinter import simpledialog
import time
from scipy.cluster.hierarchy import linkage, fcluster
from math import sqrt
from farthest_fisrt import Farthest_First_Traversal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')  # Important pour la compatibilité
from matplotlib.figure import Figure  # Import manquant qui cause l'erreur
from tkinter import *
import tkinter as tk
from tkinter import ttk  # Ajoutez cet import en haut du fichier
from scipy.cluster.hierarchy import cut_tree
import networkx as nx
import itertools
class Espace_metrique:

    def __init__(self, parent,taille_axe_x=10.0,taille_axe_y=10.0):

        self.parent = parent
        self.taille_axe_x = taille_axe_x
        self.taille_axe_y = taille_axe_y  

        self.canvas_width = 800
        self.canvas_height = 600
        self.marge_gauche = 50
        self.marge_droite = 50
        self.marge_haut = 50
        self.marge_bas = 50

        self.x_scale = (self.canvas_width - self.marge_gauche - self.marge_droite) / self.taille_axe_x
        self.y_scale = (self.canvas_height - self.marge_haut - self.marge_bas) / self.taille_axe_y
        self.x_min = self.marge_gauche
        self.y_max = self.canvas_height - self.marge_bas  # L'axe Y est inversé en pixels
       
       
        self.canvas=Canvas
        self.points=[]
        self.CoûtOptimal=0
        self.k_entry=0
        self.frame_contenu=Frame(parent)
        self.canvas = Canvas(parent, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.canvas.pack(pady=10)
        self.selected_point = None
        self.n_max = 0
        self.lettres = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        
        self.tracer_axes()

        self.canvas.bind("<Button-1>", self.traiter_clic)
        self.canvas.bind("<B1-Motion>", self.decaler_point)
        self.canvas.bind("<ButtonRelease-1>", self.deselectionner_point)
        
        # Frame pour les résultats
        self.frame_resultats = Frame(parent)
        self.frame_resultats.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Zone de texte pour les résultats
        self.text_resultats = Text(self.frame_resultats, height=15, wrap=WORD)
        self.scrollbar = Scrollbar(self.frame_resultats, command=self.text_resultats.yview)
        self.text_resultats.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side=RIGHT, fill=Y)
        self.text_resultats.pack(side=LEFT, fill=BOTH, expand=True)

        self.selected_indices = []  # Initialisation ajoutée
        self.R_values = []
        self.parent_indices = []
        self.positions_pi = None


    def ajouter_point(self, point):
        """Ajoute un point à l'espace"""
        self.points.append(point)
    
    #def dessiner_point(self, point):
        """Dessine un point sur le canvas"""
       # x, y = point.x, point.y
        #self.canvas.create_oval(x-3, y-3, x+3, y+3, fill='red', outline='black')
        #self.canvas.create_text(x+10, y+10, text=str(point.nombre), fill='black')

    def dessiner_point(self):
        self.canvas.delete("point")  # Supprime les anciens points (facultatif)
        for p in self.points:
            x, y = self.coord_to_pixel(p.x, p.y)  # Conversion selon l’échelle
            r = 5
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='blue', tags="point")

    def traiter_clic_simule(self, x, y):
        """Simule un clic pour ajouter un point"""
        point = Point(x, y, len(self.points)+1)
        self.points.append(point)
        self.dessiner_point(point)

    def effacer_tous_les_points(self):
        """Efface tous les points de l'espace métrique"""
        # Supprime tous les points de la liste
        self.points = []
        # Supprime tous les éléments du canvas
        self.canvas.delete("all")
        # Réinitialise les axes
        self.dessiner_axes()

    def dessiner_axes(self):
        """Dessine les axes x et y de manière visible"""
        # Effacer les anciens axes
        self.canvas.delete("axe")
    
        # Couleur et épaisseur des axes
        axe_color = "black"
        axe_width = 2
    
        # Dessiner l'axe X
        self.canvas.create_line(
            0, self.taille_axe_y, 
            self.taille_axe_x, self.taille_axe_y,
            fill=axe_color, width=axe_width, tags="axe"
        )
    
        # Dessiner l'axe Y
        self.canvas.create_line(
            0, 0, 
            0, self.taille_axe_y,
            fill=axe_color, width=axe_width, tags="axe"
        )
    
        # Ajouter des étiquettes
        self.canvas.create_text(
            self.taille_axe_x/2, self.taille_axe_y + 20,
            text="Axe X", fill=axe_color, tags="axe"
        )
        self.canvas.create_text(
            20, self.taille_axe_y/2,
            text="Axe Y", fill=axe_color, angle=90, tags="axe"
        )
        
    def setup_ui(self):
        """Configuration de l'interface utilisateur optimisée pour macOS"""
        #self.frame = ttk.Frame(self.parent)
        #self.canvas = tk.Canvas(self.frame, width=800, height=600, bg='white')
        #self.canvas.pack(pady=10)
        #self.canvas.bind("<Button-1>", self.traiter_clic)
        #self.frame.pack()

    def afficher(self):
        self.frame_contenu.pack(pady=10)
        #self.canvas.pack(expand=True, fill=BOTH)
        #self.frame.pack(expand=True, fill=BOTH) 

    def afficher_clusters(self, points_array, clusters, k):
        """Affiche visuellement les clusters avec couleurs différentes"""
        fenetre = Toplevel(self.parent)
        fenetre.title(f"Visualisation des {k} Clusters")
    
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    
    # Tracer les points avec des couleurs par cluster
        scatter = ax.scatter(points_array[:,0], points_array[:,1], 
                        c=clusters, cmap='tab20', s=100)
    
    # Ajouter les étiquettes des points (A, B, C...)
        for i, point in enumerate(self.points):
            ax.text(point[5]+0.02, point[6]+0.02, 
                point[4], fontsize=12, color='black')
    
    # Ajouter une légende
        ax.set_title(f"Résultat du Clustering (k={k})")
        ax.grid(True)
    
    # Intégration dans Tkinter
        canvas = FigureCanvasTkAgg(fig, master=fenetre)
        canvas.draw()
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

    #----- cette methode pour dessigner les axes x et y avec les quadre ---------
    def tracer_axes(self):

        # Efface tout d'abord
        self.canvas.delete("all")
    
        # Zone de dessin
        self.x_min, self.x_max = self.marge_gauche, self.canvas_width - self.marge_droite
        self.y_min, self.y_max = self.marge_haut, self.canvas_height - self.marge_bas
        
        # Calcul des échelles
        self.x_scale = (self.x_max - self.x_min) / self.taille_axe_x
        self.y_scale = (self.y_max - self.y_min) / self.taille_axe_y

        self.x_min = self.x_min
        self.y_max = self.y_max  # Note: l'axe Y est inversé en pixels
        # Création du quadrillage
        for x in range(0, int(self.taille_axe_x) + 1):
            x_pixel = self.x_min + x * self.x_scale
            self.canvas.create_line(x_pixel, self.y_min, x_pixel, self.y_max, fill='lightgray', dash=(2,2))
        
        for y in range(0, int(self.taille_axe_y) + 1):
            y_pixel = self.y_max - y * self.y_scale
            self.canvas.create_line(self.x_min, y_pixel, self.x_max, y_pixel, fill='lightgray', dash=(2,2))
    
        # Axe X avec flèche
        self.canvas.create_line(self.x_min, self.y_max, self.x_max, self.y_max, width=2, arrow=LAST)
        # Axe Y avec flèche
        self.canvas.create_line(self.x_min, self.y_max, self.x_min, self.y_min, width=2, arrow=LAST)
        
        # Graduations et étiquettes axe X
        for x in range(0, int(self.taille_axe_x) + 1):
            x_pixel = self.x_min + x * self.x_scale
            self.canvas.create_line(x_pixel, self.y_max, x_pixel, self.y_max+5, width=1)
            self.canvas.create_text(x_pixel, self.y_max+15, text=str(x), 
                                font=('Arial', 8), anchor=N)
    
        # Graduations et étiquettes axe Y
        for y in range(0, int(self.taille_axe_y) + 1):
            y_pixel = self.y_max - y * self.y_scale
            self.canvas.create_line(self.x_min-5, y_pixel, self.x_min, y_pixel, width=1)
            self.canvas.create_text(self.x_min-10, y_pixel, text=str(y), 
                                font=('Arial', 8), anchor=E)
        # Étiquettes des axes
        self.canvas.create_text(self.x_max+10, self.y_max, text="X", font=('Arial', 10))
        self.canvas.create_text(self.x_min, self.y_min-10, text="Y", font=('Arial', 10))



    #---- cette methode pour generer l'ajoute et la selection des points -----
    def traiter_clic(self, event):
    
        if not (self.x_min <= event.x <= (self.canvas_width - self.marge_droite) and 
                self.marge_haut <= event.y <= (self.canvas_height - self.marge_bas)):
            return
        # Vérifier si on clique sur un point existant
        for i, point in enumerate(self.points):
            x_pixel, y_pixel = point[0], point[1]
            if sqrt((event.x - x_pixel)**2 + (event.y - y_pixel)**2) <= 10:
                if self.selected_point is not None:
                    self.canvas.itemconfig(self.points[self.selected_point][2], fill='red')
                self.selected_point = i
                self.canvas.itemconfig(point[2], fill='green')
                return
    
        if len(self.points) >= self.n_max:
            messagebox.showinfo("Attention", f"Vous avez déjà sélectionné {self.n_max} points.")
            return
        
        # Conversion des coordonnées en valeurs mathématiques
        x_math = (event.x - self.x_min) / self.x_scale
        y_math = (self.y_max - event.y) / self.y_scale

            
        lettre = self.lettres[len(self.points)]
        point_id = self.canvas.create_oval(event.x-5, event.y-5, event.x+5, event.y+5, 
                                             fill='red', outline='black')
        text_id = self.canvas.create_text(event.x, event.y+15, 
                                            text=f"{lettre}({x_math:.2f},{y_math:.2f})",
                                            font=('Arial', 10), fill='black')
            
        self.points.append((event.x, event.y, point_id, text_id, lettre, x_math, y_math))
        self.selected_point = len(self.points) - 1
        self.canvas.itemconfig(point_id, fill='green')
        


    # ----- cette methodes pour deplacer ou changer les coordonner d'un point 
    def set_dimensions(self, taille_axe_x, taille_axe_y):
        """Met à jour les dimensions de l'espace métrique"""
        self.taille_axe_x = float(taille_axe_x)
        self.taille_axe_y = float(taille_axe_y)
        self.tracer_axes()  # Redessine les axes avec les nouvelles dimensions
    
    def decaler_point(self, event):
        if self.selected_point is not None:
            idx = self.selected_point
            x_pixel = max(self.x_min, min(event.x, self.canvas_width - self.marge_droite))
            y_pixel = max(self.marge_haut, min(event.y, self.canvas_height - self.marge_bas))
            
            x_math = round((x_pixel - self.x_min) / self.x_scale)
            y_math = round((self.y_max - y_pixel) / self.y_scale)
            
            self.canvas.coords(self.points[idx][2], x_pixel-5, y_pixel-5, x_pixel+5, y_pixel+5)
            self.canvas.coords(self.points[idx][3], x_pixel, y_pixel+15)
            self.canvas.itemconfig(self.points[idx][3], 
                                 text=f"{self.points[idx][4]}({x_math:.2f},{y_math:.2f})")
            self.points[idx] = (x_pixel, y_pixel, self.points[idx][2], self.points[idx][3], 
                               self.points[idx][4], x_math, y_math)
            
    #--- pour deselectionner un point dans l'espace 
    def deselectionner_point(self, event):
        pass


    #---- ce methode supprimer un point qui l'utilisateur selectioner dans l'espace 
    def retirer_point_selectionne(self):
        if self.selected_point is not None:
            idx = self.selected_point
            
            # Supprime les éléments graphiques
            self.canvas.delete(self.points[idx][2])
            self.canvas.delete(self.points[idx][3])
            
            # Supprime le point de la liste
            self.points.pop(idx)
            self.selected_point = None
            
            # Met à jour les lettres des points restants
            for i, point in enumerate(self.points):
                x_pixel, y_pixel, point_id, text_id, _, x_math, y_math = point
                lettre = self.lettres[i]
                self.canvas.itemconfig(text_id, text=f"{lettre}({x_math:.2f},{y_math:.2f})")
                self.points[i] = (x_pixel, y_pixel, point_id, text_id, lettre, x_math, y_math)
        else:
            messagebox.showwarning("Aucun point sélectionné", "Veuillez d'abord cliquer sur un point à supprimer")
    
    # cet methode pour reinitailier tous les points qui se trouve dans l'espace metrique
    def reinitialiser_espace(self, n_max):
        
        self.n_max = min(n_max, len(self.lettres))
        self.points = []
        self.selected_point = None
        self.canvas.delete("all")
        self.tracer_axes()
        self.text_resultats.delete(1.0, END)

    def k_center_optimal(self,points_dict, k):
        # Préparation des données à partir du dictionnaire
        labels = list(points_dict.keys())
        coords = np.array([points_dict[i] for i in labels])
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        dist_matrix = distance_matrix(coords, coords)

        def max_radius(clusters, centers):
            max_dist = 0
            farthest_points = {}
            for cidx, cluster in enumerate(clusters):
                center = centers[cidx]
                center_idx = label_to_index[center]
                cluster_max_dist = 0
                farthest_point = None
                for p in cluster:
                    dist = dist_matrix[label_to_index[p]][center_idx]
                    if dist > cluster_max_dist:
                        cluster_max_dist = dist
                        farthest_point = p
                    max_dist = max(max_dist, dist)
                farthest_points[cidx] = (farthest_point, cluster_max_dist)
            return max_dist, farthest_points

        # Générateur de partitions
        def all_partitions_k(elements, k):
            def helper(parts, remaining):
                if len(parts) > k:
                    return
                if not remaining:
                    if len(parts) == k:
                        yield parts
                    return
                for i in range(len(parts)):
                    yield from helper(parts[:i] + [parts[i] + [remaining[0]]] + parts[i+1:], remaining[1:])
                yield from helper(parts + [[remaining[0]]], remaining[1:])
            return helper([], elements)

        # Algorithme principal
        best_clusters = None
        best_centers = None
        min_cost = float('inf')
        best_farthest_points = None
                    
        for partition in all_partitions_k(labels, k):
            if any(len(c) == 0 for c in partition):
                continue
            for possible_centers in itertools.product(*partition):
                cost, farthest_points = max_radius(partition, possible_centers)
                if cost < min_cost:
                    min_cost = cost
                    best_clusters = partition
                    best_centers = possible_centers
                    best_farthest_points = farthest_points
        return best_clusters, best_centers, min_cost, best_farthest_points
                

    def creer_arbre_pi(self):
        """Crée l'arbre π avec l'algorithme Farthest-First Traversal exact"""
        try:
            if len(self.points) < 3:
                messagebox.showerror("Erreur", "Au moins 3 points sont nécessaires")
                return

        # Conversion des points en array numpy
            points_array = np.array([[p[5], p[6]] for p in self.points])
            self.positions_pi = points_array.copy()  # Stockage pour cohérence
        # Calcul de la matrice de distance
            D = distance_matrix(points_array, points_array)
            n_points = len(points_array)
        
        # Initialisation FFT (comme dans votre exemple)
            remaining_indices = list(range(n_points))
            first_index = np.random.choice(remaining_indices)

            self.selected_indices = [first_index]  # Initialisation de l'attribut
            remaining_indices.remove(first_index)
            self.R_values = [np.inf]  # Initialisation de l'attribut
            self.parent_indices = [None]  # Initialisation de l'attribut

        # Construction de l'arbre
            for step in range(1, n_points):
                dists_to_selected = D[:, self.selected_indices]
                min_dists = np.min(dists_to_selected, axis=1)
                min_dists[self.selected_indices] = -1  # Ignorer les points déjà sélectionnés
                next_index = np.argmax(min_dists)
                Ri = min_dists[next_index]
                self.R_values.append(Ri)
            # Trouver le parent le plus proche
                parent_distances = D[next_index, self.selected_indices]
                parent_index = self.selected_indices[np.argmin(parent_distances)]
                self.parent_indices.append(parent_index)
            
            # Mise à jour des listes
                self.selected_indices.append(next_index)
                remaining_indices.remove(next_index)


            edges = [(self.selected_indices[k], self.parent_indices[k]) for k in range(1, n_points)]
            distances = self.R_values[1:]

            self._afficher_arbre(
                points_array=points_array,
                edges=edges,
                distances=distances,
                title="Arbre π (Farthest-First Traversal)",
                line_style='--'
            )

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur création arbre π: {str(e)}")


    def _afficher_arbre(self, points_array, edges, distances, title, line_style):
        """Méthode interne pour l'affichage cohérent des arbres avec option de clustering"""
        fenetre = Toplevel(self.parent)
        fenetre.title(title)
    
    # Frame principale
        main_frame = Frame(fenetre)
        main_frame.pack(fill=BOTH, expand=True)
    
    # Frame pour le contrôle k - plus visible
        control_frame = Frame(main_frame, bd=2, relief=GROOVE, padx=10, pady=10)
        control_frame.pack(side=TOP, fill=X, padx=5, pady=5)
    
    # Titre pour la section clustering
        #Label(control_frame, 
            #text="Clustering par suppression d'arêtes",
            #font=('Arial', 10, 'bold')).pack(pady=5)
    
    # Frame pour le champ de saisie et le bouton
        #control_frame = Frame(control_frame)
        #control_frame.pack(pady=5)
    
    # Ajout du champ de saisie pour k avec un label plus clair
        Label(control_frame, text="Nombre de clusters (k):").pack(side=LEFT)
        self.k_entry = Entry(control_frame, width=5)
        self.k_entry.pack(side=LEFT, padx=5)
    # Bouton "OK" plus visible
        btn_show = Button(control_frame, 
                        text="Afficher Clusters", 
                        bg="#4CAF50",  # Vert
                        fg="royal blue",
                        width=8,
                        command=lambda: self._appliquer_clustering(
                            points_array, 
                            edges, 
                            distances, 
                            int(self.k_entry.get()) if self.k_entry.get().isdigit() else 0
                        ))
        btn_show.pack(side=LEFT, padx=5)
    
    #ici pour le buton de calcule le cout :
        btn_calc = Button(control_frame, 
            text="Calculer Coût meilleur k-clustering", 
            bg="#2196F3",  # Bleu
            fg="royal blue",
            width=15,
            command=lambda: self._afficher_resultat_cout(
                points_array,
                [(edges[i][0],edges[i][1], distances[i]) for i in range(len(edges))],
              int(self.k_entry.get()) if self.k_entry.get().isdigit() else 0
            ))
        btn_calc.pack(side=LEFT, padx=5)
    
    # Message d'information
        Label(control_frame, 
            text=f"Entrez un entier entre 1 et {len(edges)+1}",
            font=('Arial', 8)).pack(pady=2)
    
    # Figure matplotlib
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    
    # Points
        ax.scatter(points_array[:, 0], points_array[:, 1], c='gray', s=60)
    
    # Arêtes
        for i, ((child_idx, parent_idx), dist) in enumerate(zip(edges, distances)):
            start = points_array[parent_idx]
            end = points_array[child_idx]
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                f'k{line_style}', linewidth=1)
        
        # Distance
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, f"{dist:.2f}", fontsize=9, color='blue')
    
    # Numérotation
        for k, idx in enumerate(self.selected_indices):
            ax.scatter(points_array[idx][0], points_array[idx][1], c='red', s=80)
            ax.text(points_array[idx][0]+0.03, points_array[idx][1]+0.03,
            str(k+1), fontsize=12, color='red')
    
        ax.grid(True)
        ax.set_aspect('equal')
    
        canvas = FigureCanvasTkAgg(fig, master=main_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    
    # Stocker les données pour le clustering
        self._current_tree_data = {
            'points_array': points_array,
            'edges': edges,
            'distances': distances
        }

    def _appliquer_clustering(self, points_array, edges, distances, k):
        """Applique le clustering en conservant la numérotation originale"""
        try:
            if k < 1 or k > len(self.selected_indices):
                messagebox.showerror("Erreur", f"k doit être entre 1 et {len(self.selected_indices)}")
                return
    
            # Créer la liste des arêtes avec distances
            edges_with_dist = [(child, parent, dist) for (child, parent), dist in zip(edges, distances)]
    
            # Trier les arêtes par distance décroissante
            edges_sorted = sorted(edges_with_dist, key=lambda x: x[2], reverse=True)
    
            # Nombre d'arêtes à supprimer = k-1
            edges_to_remove = min(k-1, len(edges_sorted))
            edges_to_keep = edges_sorted[edges_to_remove:]
    
            # Créer une nouvelle fenêtre pour afficher le résultat
            cluster_fen = Toplevel(self.parent)
            cluster_fen.title(f"Clustering avec k={k}")
    
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
    
            # Calculer les clusters
            clusters = self._calculer_clusters(points_array, edges_to_keep)
    
            # Afficher les points avec leurs couleurs de cluster
            scatter = ax.scatter(points_array[:, 0], points_array[:, 1], c=clusters, cmap='tab20', s=80)
    
            # Afficher les arêtes restantes
            for child, parent, dist in edges_to_keep:
                start = points_array[parent]
                end = points_array[child]
                ax.plot([start[0], end[0]], [start[1], end[1]], 'k--', linewidth=1)
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                ax.text(mid_x, mid_y, f"{dist:.2f}", fontsize=9, color='blue')
    
            # Numérotation originale
            for idx in self.selected_indices:
                ax.text(points_array[idx][0]+0.03, points_array[idx][1]+0.03,
                    str(self.selected_indices.index(idx)+1), 
                    fontsize=12, color='red')
    
            ax.set_title(f"Clustering (k={k})")
            ax.grid(True)
    
            canvas = FigureCanvasTkAgg(fig, master=cluster_fen)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du clustering: {str(e)}")

    def _calculer_clusters(self, points_array, edges_to_keep):
        """Calcule les clusters à partir des arêtes conservées"""
        # Union-Find pour trouver les composantes connexes
        parent = [i for i in range(len(points_array))]
    
        def find(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
    
        # Connecter les points selon les arêtes conservées
        for child, parent_idx, dist in edges_to_keep:
            root_child = find(child)
            root_parent = find(parent_idx)
            if root_child != root_parent:
                parent[root_child] = root_parent
    
        # Assigner les numéros de cluster
        clusters = np.zeros(len(points_array), dtype=int)
        cluster_id = 1
        cluster_ids = {}
    
        for i in range(len(points_array)):
            root = find(i)
            if root not in cluster_ids:
                cluster_ids[root] = cluster_id
                cluster_id += 1
            clusters[i] = cluster_ids[root]
    
        return clusters

    #creation ou la construction de l'arbre hiérarchique T^pi'
    def creer_arbre_pi_prime(self, alpha=1, beta=2):
        """Crée l'arbre π' avec la bonne hiérarchie de granularité"""
        try:
            # Vérifications initiales
            if not hasattr(self, 'selected_indices'):
                messagebox.showerror("Erreur", "Veuillez d'abord créer l'arbre π")
                return

            points_array = self.positions_pi.copy()
            n_points = len(points_array)
        
        # Initialisation des structures
            point_levels = [0] * n_points  # Niveau 0 pour tous initialement
            granularity_levels = {0: [1]}  # L_0 contient toujours juste la racine

        # Calcul de la matrice de distance
            D = distance_matrix(points_array, points_array)
        
        # Calcul de R
            if len(self.R_values) < 2:
                messagebox.showerror("Erreur", "Pas assez de valeurs R disponibles")
                return
            R = alpha * self.R_values[1]  # R = α * R₂

            # Nouvel algorithme de calcul des niveaux
            for i in range(1, n_points):
                Ri = self.R_values[i]
                level = 0
            
            # Condition inversée pour un calcul correct
                while Ri <= R / (beta ** level):
                    level += 1
            
            # Le niveau est égal au nombre de divisions par β nécessaires
                point_level = level
            
            # Assignation du niveau
                point_levels[i] = point_level
            
            # Ajout au niveau correspondant
                if point_level not in granularity_levels:
                    granularity_levels[point_level] = []
                granularity_levels[point_level].append(i + 1)  # +1 pour la numérotation

        # Construction de π' avec la bonne hiérarchie
            pi_prime = {}
            edge_lengths = {}

            for i in range(1, n_points):
                current_level = point_levels[i]
                child_num = i + 1
            
            # Parents potentiels (niveaux strictement inférieurs)
                candidates = []
                for lvl in range(current_level):
                    candidates.extend(granularity_levels.get(lvl, []))
            
            # Trouver le parent le plus proche
                min_dist = np.inf
                closest_parent = 1  # Racine par défaut
            
                for candidate in candidates:
                    dist = D[self.selected_indices[i], self.selected_indices[candidate-1]]
                    if 0 < dist < min_dist:
                        min_dist = dist
                        closest_parent = candidate
            
                pi_prime[child_num] = closest_parent
                edge_lengths[(child_num, closest_parent)] = min_dist

        # Vérification finale de la hiérarchie
            for child, parent in pi_prime.items():
                child_level = point_levels[child-1]
                parent_level = point_levels[parent-1] if parent != 1 else 0
            
                if parent_level >= child_level:
                # Correction automatique si nécessaire
                    pi_prime[child] = 1
                    edge_lengths[(child, 1)] = D[self.selected_indices[child-1], self.selected_indices[0]]

        # Création du texte avec la bonne structure
            info_text = f"Valeur de R (α × R₂) = {R:.4f}\n\n"
            info_text += "Niveaux de granularité :\n"
        
        # Tri des niveaux pour l'affichage
            for lvl in sorted(granularity_levels.keys()):
                info_text += f"L_{lvl} : {sorted(granularity_levels[lvl])}\n"
        
            info_text += "\nCorrespondance enfant → parent :\n"
            for child in sorted(pi_prime.keys()):
                info_text += f"π'({child}) = {pi_prime[child]}\n"

        # Création de la fenêtre avec les contrôles de clustering
            fenetre = Toplevel(self.parent)
            fenetre.title(f"Arbre π' (α={alpha}, β={beta})")
            fenetre.geometry("900x700")
        
            # Frame principale
            main_frame = Frame(fenetre)
            main_frame.pack(fill=BOTH, expand=True)
        
            # Frame pour les contrôles
            control_frame = Frame(main_frame, bd=2, relief=GROOVE, padx=10, pady=10)
            control_frame.pack(side=TOP, fill=X, padx=5, pady=5)
        
            # Champ de saisie pour k
            Label(control_frame, text="Nombre de clusters (k):").pack(side=LEFT)
            self.k_entry = Entry(control_frame, width=5)
            self.k_entry.pack(side=LEFT, padx=5)
        
            # Bouton Afficher Clusters
            btn_show = Button(control_frame,
                            text="Afficher Clusters",
                            bg="#4CAF50",
                            fg="royal blue",
                            width=15,
                            command=lambda: self._appliquer_clustering_pi_prime(
                                points_array,
                                [(self.selected_indices[c-1], self.selected_indices[p-1]) for c,p in pi_prime.items()],
                                [edge_lengths[(c,p)] for c,p in pi_prime.items()],
                                int(self.k_entry.get()) if self.k_entry.get().isdigit() else 0
                            ))
            
            btn_show.pack(side=LEFT, padx=5)
        
            # Bouton Calculer Coût
            btn_calc = Button(control_frame,
                            text="Calculer Coût",
                            bg="#2196F3",
                            fg="royal blue",
                            width=15,
                            command=lambda: self._afficher_resultat_cout(
                                points_array,
                                [(self.selected_indices[c-1], self.selected_indices[p-1], edge_lengths[(c,p)]) for c,p in pi_prime.items()],
                                int(self.k_entry.get()) if self.k_entry.get().isdigit() else 0
                            ))
            btn_calc.pack(side=LEFT, padx=5)

                    # Bouton Quitter
            # Message d'information
            max_k = len(pi_prime.items())
            Label(control_frame,
                text=f"Entrez un entier entre 1 et {max_k}",
                font=('Arial', 8)).pack(pady=2)
        
            # Frame pour le graphique et les infos
            content_frame = Frame(main_frame)
            content_frame.pack(fill=BOTH, expand=True)
        
            # Frame pour le graphique
            graph_frame = Frame(content_frame)
            graph_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)
        
            # Graphique matplotlib
            fig = Figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
        
            # Points
            ax.scatter(points_array[:, 0], points_array[:, 1], c='gray', s=60)
        
            # Arêtes
            for (child_idx, parent_idx), dist in zip(
                [(self.selected_indices[c-1], self.selected_indices[p-1]) for c,p in pi_prime.items()],
                [edge_lengths[(c,p)] for c,p in pi_prime.items()]
            ):
                start = points_array[parent_idx]
                end = points_array[child_idx]
            
                if not np.isfinite(dist) or dist <= 0:
                    dist = np.linalg.norm(start-end)
                
                ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1)
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                ax.text(mid_x, mid_y, f"{dist:.2f}", fontsize=9, color='blue')
        
            # Numérotation
            for k, idx in enumerate(self.selected_indices):
                ax.scatter(points_array[idx][0], points_array[idx][1], c='red', s=80)
                ax.text(points_array[idx][0]+0.03, points_array[idx][1]+0.03,
                str(k+1), fontsize=12, color='red')
        
            ax.grid(True)
            ax.set_aspect('equal')
        
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)
        
            # Frame pour les informations textuelles
            info_frame = Frame(content_frame, bd=2, relief=GROOVE)
            info_frame.pack(side=BOTTOM, fill=BOTH, padx=10, pady=10)
        
            # Zone de texte pour les informations
            text_widget = Text(info_frame, height=10, wrap=WORD)
            scrollbar = Scrollbar(info_frame, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)
        
            scrollbar.pack(side=RIGHT, fill=Y)
            text_widget.pack(side=LEFT, fill=BOTH, expand=True)
        
            # Insertion du texte d'information
            text_widget.insert(END, info_text)
            text_widget.configure(state=DISABLED)
        
            # Stocker les données pour le clustering
            self._current_tree_data = {
                'points_array': points_array,
                'edges': [(self.selected_indices[c-1], self.selected_indices[p-1]) for c,p in pi_prime.items()],
                'distances': [edge_lengths[(c,p)] for c,p in pi_prime.items()]
            }
            self._current_pi_prime_data = {
                'points_array': points_array,
                'edges_with_dist': [(self.selected_indices[c-1], self.selected_indices[p-1], edge_lengths[(c,p)]) 
                            for c,p in pi_prime.items()]
            }
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur création arbre π': {str(e)}")


    def _appliquer_clustering_pi_prime(self, points_array, edges, distances, k):
        """Applique le clustering sur l'arbre π'"""
        try:
            if k < 1 or k > len(edges)+1:
                messagebox.showerror("Erreur", f"k doit être entre 1 et {len(edges)+1}")
                return
    
            # Créer la liste des arêtes avec distances
            edges_with_dist = [(edges[i][0], edges[i][1], distances[i]) for i in range(len(edges))]
    
            # Trier les arêtes par distance décroissante
            edges_sorted = sorted(edges_with_dist, key=lambda x: x[2], reverse=True)
    
            # Nombre d'arêtes à supprimer
            edges_to_remove = min(k-1, len(edges_sorted))
            edges_to_keep = edges_sorted[edges_to_remove:]
    
            # Créer une nouvelle fenêtre
            cluster_fen = Toplevel(self.parent)
            cluster_fen.title(f"Clustering π' avec k={k}")
    
            fig = Figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
    
            # Calculer les clusters
            clusters = self._calculer_clusters(points_array, edges_to_keep)
    
            # Afficher les points avec couleurs de cluster
            scatter = ax.scatter(points_array[:, 0], points_array[:, 1], c=clusters, cmap='tab20', s=80)
    
            # Afficher les arêtes restantes
            for child, parent, dist in edges_to_keep:
                start = points_array[parent]
                end = points_array[child]
                ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', linewidth=1)
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                ax.text(mid_x, mid_y, f"{dist:.2f}", fontsize=9, color='blue')
    
            # Numérotation originale
            for idx in self.selected_indices:
                ax.text(points_array[idx][0]+0.03, points_array[idx][1]+0.03,
                    str(self.selected_indices.index(idx)+1),
                    fontsize=12, color='red')
    
            ax.set_title(f"Clustering π' (k={k})")
            ax.grid(True)
    
            canvas = FigureCanvasTkAgg(fig, master=cluster_fen)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur clustering π': {str(e)}")
            
    def _afficher_arbre_pi_prime_complet(self, points_array, edges, distances, title, line_style, info_text):
        """Affichage complet avec informations de granularité"""
        fenetre = Toplevel(self.parent)
        fenetre.title(title)
        fenetre.geometry("900x700")  # Taille augmentée pour accommoder le texte
    
        # Frame principale avec deux sous-frames
        main_frame = Frame(fenetre)
        main_frame.pack(fill=BOTH, expand=True)
    
        # Frame pour le graphique (en haut)
        graph_frame = Frame(main_frame)
        graph_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)
    
        # Frame pour les informations textuelles (en bas)
        info_frame = Frame(main_frame, bd=2, relief=GROOVE)
        info_frame.pack(side=BOTTOM, fill=BOTH, padx=10, pady=10)
    
        # Graphique matplotlib
        fig = Figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
    
        # Points
        ax.scatter(points_array[:, 0], points_array[:, 1], c='gray', s=60)
    
        # Arêtes
        for (child_idx, parent_idx), dist in zip(edges, distances):
            start = points_array[parent_idx]
            end = points_array[child_idx]

            if not np.isfinite(dist) or dist <= 0:
                dist=np.linalg.norm(start-end) 

            ax.plot([start[0], end[0]], [start[1], end[1]], 
                f'k{line_style}', linewidth=1)
        
            # Distance
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, f"{dist:.2f}", fontsize=9, color='blue')
    
        # Numérotation
        for k, idx in enumerate(self.selected_indices):
            ax.scatter(points_array[idx][0], points_array[idx][1], c='red', s=80)
            ax.text(points_array[idx][0]+0.03, points_array[idx][1]+0.03,
                str(k+1), fontsize=12, color='red')
    
        ax.grid(True)
        ax.set_aspect('equal')
    
        canvas = FigureCanvasTkAgg(fig, master=graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)
    
        # Zone de texte pour les informations
        text_widget = Text(info_frame, height=10, wrap=WORD)
        scrollbar = Scrollbar(info_frame, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
    
        scrollbar.pack(side=RIGHT, fill=Y)
        text_widget.pack(side=LEFT, fill=BOTH, expand=True)
    
        # Insertion du texte d'information
        text_widget.insert(END, info_text)
        text_widget.configure(state=DISABLED)  # Rend le texte en lecture seule

    def _calculer_cout_clustering(self, points_array, edges, distances, k):
        """Calcule et affiche le coût du clustering"""
        try:
            if k < 1 or k > len(edges)+1:
                messagebox.showerror("Erreur", f"k doit être entre 1 et {len(edges)+1}")
                return
        
        # Préparer les données pour cout_clustering_arbre
            edges_with_dist = [(edges[i][0], edges[i][1], distances[i]) for i in range(len(edges))]
        
        # Appeler la fonction de calcul de coût
            max_diameter, point_pair, cluster_index = Clustering.cout_clustering_arbre(
                points_array, 
                edges_with_dist,
                self.selected_indices,
                k
            )
        
        # Afficher les résultats dans une fenêtre
            result_fen = Toplevel(self.parent)
            result_fen.title(f"Coût du {k}-clustering")
        
            text = Text(result_fen, height=8, width=50)
            text.pack(padx=10, pady=10)
        
            text.insert(END, f"Résultats pour k={k} clusters:\n\n")
            text.insert(END, f"Coût (diamètre maximal): {max_diameter:.4f}\n")
            text.insert(END, f"Points réalisant ce diamètre: {point_pair[0]} et {point_pair[1]}\n")
            text.insert(END, f"Cluster concerné: {cluster_index}\n\n")
        
        # Bouton pour fermer
            Button(result_fen, 
                text="Fermer", 
                command=result_fen.destroy).pack(pady=5)
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur dans le calcul du coût: {str(e)}")

    def calculer_CH(self):
        try:

            if len(self.points) < 2:
                messagebox.showerror("Erreur", "Au moins 2 points sont nécessaires")
                return
            
            k = simpledialog.askinteger("Clustering", 
                                  "Entrez le nombre de clusters souhaité (k):",
                                  parent=self.parent,
                                  minvalue=1,
                                  maxvalue=len(self.points))
            if not k:
                return
            
            points_array = np.array([[p[5], p[6]] for p in self.points])
            Z=linkage(points_array,method='ward')
            clusters = fcluster(Z, k, criterion='maxclust')


            #self.afficher_clusters(points_array, clusters,k)
            self._afficher_resultats_clustering(points_array, Z, clusters, k)

            #Clustering.clustering_hierarchique(self,points_objects)
        except Exception as e:
            messagebox.showerror("Erreur",f"Erreur lors  du clustering:{str(e)}")

    #def calculer_facteur(self):


    def cout_clustering_arbre(self, points, edges_with_dist, selected_indices, nb_clusters):
        """Calcule le coût du clustering selon votre algorithme"""
        try:
            # Construction du graphe initial avec networkx
            G = nx.Graph()
            for (child, parent, dist) in edges_with_dist:
                # Conversion des indices
                num_child = selected_indices.index(child) + 1
                num_parent = selected_indices.index(parent) + 1
                G.add_edge(num_child, num_parent, weight=dist)

            # Tri des arêtes par distance décroissante
            edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

            # Suppression des (nb_clusters-1) arêtes les plus longues
            for i in range(nb_clusters-1):
                if i < len(edges_sorted):
                    G.remove_edge(*edges_sorted[i][:2])

            # Récupération des clusters
            components = list(nx.connected_components(G))
        
            # Calcul du diamètre maximal
            max_diameter = 0
            point_pair = (None, None)
            cluster_index = None

            for idx, cluster in enumerate(components, 1):
                cluster_points = [points[selected_indices[num-1]] for num in cluster]
                if len(cluster_points) >= 2:
                    D = distance_matrix(cluster_points, cluster_points)
                    current_max = np.max(D)
                    if current_max > max_diameter:
                        max_diameter = current_max
                        # Trouver la paire de points
                        i, j = np.unravel_index(np.argmax(D), D.shape)
                        cluster_list = list(cluster)
                        point_pair = (cluster_list[i], cluster_list[j])
                        cluster_index = idx

            return max_diameter, point_pair, cluster_index

        except Exception as e:
            raise Exception(f"Erreur dans cout_clustering_arbre: {str(e)}")
    
    def _afficher_resultat_cout(self, points_array, edges_with_dist, k):
        """Affiche les résultats du calcul de coût"""
        try:
            if k < 1 or k > len(edges_with_dist)+1:
                messagebox.showerror("Erreur", f"k doit être entre 1 et {len(edges_with_dist)+1}")
                return
        
            # Appel CORRIGÉ de la méthode
            max_diameter, point_pair, cluster_index =self.cout_clustering_arbre(
                points_array, 
                edges_with_dist, 
                self.selected_indices, 
                k
            )
            result_data = {
            'k': k,
            'cost': max_diameter,
            'points': point_pair,
            'cluster': cluster_index,
            'time': time.time()
            }

            # Stockage des résultats selon l'arbre courant
            if hasattr(self, '_current_pi_prime_data') and edges_with_dist == self._current_pi_prime_data['edges_with_dist']:
                self._last_pi_prime_result = result_data
                
            else:
                self._last_pi_result=result_data
                
            # Création de la fenêtre de résultats
            result_fen = Toplevel(self.parent)
            result_fen.title(f"Résultats du clustering (k={k})")
        
            # Affichage détaillé
            text = Text(result_fen, height=15, width=70)
            scrollbar = Scrollbar(result_fen, command=text.yview)
            text.config(yscrollcommand=scrollbar.set)
        
            scrollbar.pack(side=RIGHT, fill=Y)
            text.pack(side=LEFT, fill=BOTH, expand=True)
        
            # Génération du rapport
            report = [
                f"=== RAPPORT DE CLUSTERING (k={k}) ===",
                f"\nCoût (diamètre maximal): {max_diameter:.4f}",
                f"Points extrêmes: {point_pair[0]} et {point_pair[1]}",
                f"Cluster concerné: {cluster_index}",
                "\nArêtes supprimées:"
            ]
        
            # Ajout des arêtes supprimées
            G = nx.Graph()
            for (child, parent, dist) in edges_with_dist:
                num_child = self.selected_indices.index(child) + 1
                num_parent = self.selected_indices.index(parent) + 1
                G.add_edge(num_child, num_parent, weight=dist)
        
            edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)
        
            for i in range(k-1):
                if i < len(edges_sorted):
                    u, v, w = edges_sorted[i]
                    report.append(f"- ({u}, {v}) : {w['weight']:.4f}")
        
            # Ajout des clusters finaux
            report.append("\nComposition des clusters:")
            for i in range(k-1):
                if i < len(edges_sorted):
                    G.remove_edge(*edges_sorted[i][:2])
        
            components = list(nx.connected_components(G))
            for idx, comp in enumerate(components, 1):
                report.append(f"Cluster {idx}: {sorted(comp)}")
        
            # Affichage final
            text.insert(END, "\n".join(report))
            text.config(state=DISABLED)
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur dans le calcul du coût: {str(e)}")
    
    def _afficher_resultats_clustering(self, points_array, edges_with_dist, k):
        """Affiche les résultats du clustering avec visualisation graphique et détails textuels"""
        try:
            # Validation des entrées
            if k < 1 or k > len(edges_with_dist)+1:
                messagebox.showerror("Erreur", f"k doit être entre 1 et {len(edges_with_dist)+1}")
                return

            # Calcul du clustering
            max_diameter, point_pair, cluster_index = self.cout_clustering_arbre(
                points_array, 
                edges_with_dist, 
                self.selected_indices, 
                k
            )

            # Stockage des résultats pour comparaison ultérieure
            if hasattr(self, '_current_pi_prime_data') and edges_with_dist == self._current_pi_prime_data['edges_with_dist']:
                self._last_pi_prime_result = {
                    'k': k,
                    'cost': max_diameter,
                    'points': point_pair,
                    'cluster': cluster_index,
                    'time': time.time()
                }
                tree_type = "π'"
            else:
                self._last_pi_result = {
                    'k': k,
                    'cost': max_diameter,
                    'points': point_pair,
                    'cluster': cluster_index,
                    'time': time.time()
                }
                tree_type = "π"

            # Création de la fenêtre
            fenetre = Toplevel(self.parent)
            fenetre.title(f"Résultats du clustering (k={k}, Arbre {tree_type})")
            fenetre.geometry("800x600")

            # Frame principal
            main_frame = Frame(fenetre)
            main_frame.pack(fill=BOTH, expand=True)

            # Frame pour le graphique
            graph_frame = Frame(main_frame)
            graph_frame.pack(side=TOP, fill=BOTH, expand=True, padx=10, pady=10)

            # Calcul des clusters
            clusters = self._calculer_clusters(points_array, edges_with_dist[:len(edges_with_dist)-(k-1)])

            # Création du graphique
            fig = Figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
        
            # Affichage des points avec couleurs par cluster
            scatter = ax.scatter(points_array[:,0], points_array[:,1], c=clusters, cmap='tab20', s=100)

            # Affichage des arêtes restantes
            for child, parent, dist in edges_with_dist[:len(edges_with_dist)-(k-1)]:
                start = points_array[parent]
                end = points_array[child]
                ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.5)
                mid_x = (start[0] + end[0]) / 2
                mid_y = (start[1] + end[1]) / 2
                ax.text(mid_x, mid_y, f"{dist:.2f}", fontsize=8, color='blue')

            # Numérotation des points
            for i, idx in enumerate(self.selected_indices):
                ax.text(points_array[idx][0]+0.05, points_array[idx][1]+0.05, 
                    str(i+1), fontsize=12, color='red')

            ax.set_title(f"Clustering (k={k}) - Arbre {tree_type}")
            ax.grid(True)

            # Intégration du graphique dans Tkinter
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=BOTH, expand=True)

            # Frame pour les résultats textuels
            text_frame = Frame(main_frame, bd=2, relief=GROOVE)
            text_frame.pack(side=BOTTOM, fill=BOTH, padx=10, pady=10)

            # Zone de texte avec scrollbar
            text_widget = Text(text_frame, height=10, wrap=WORD)
            scrollbar = Scrollbar(text_frame, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)

            scrollbar.pack(side=RIGHT, fill=Y)
            text_widget.pack(side=LEFT, fill=BOTH, expand=True)

            # Génération du rapport
            rapport = [
                f"=== RÉSULTATS DU CLUSTERING (k={k}) ===",
                f"Arbre: {tree_type}",
                f"\nCoût (diamètre maximal): {max_diameter:.4f}",
                f"Points extrêmes: {point_pair[0]+1} et {point_pair[1]+1}",
                f"Cluster concerné: {cluster_index}",
                "\nComposition des clusters:"
            ]

            # Détails des clusters
            unique_clusters = np.unique(clusters)
            for cluster_num in unique_clusters:
                points_in_cluster = np.where(clusters == cluster_num)[0]
                rapport.append(f"\nCluster {cluster_num}:")
                for idx in points_in_cluster:
                    point_idx = self.selected_indices[idx]
                    p = self.points[point_idx]
                    rapport.append(f"- Point {idx+1} ({p[4]}): ({p[5]:.2f}, {p[6]:.2f})")

            # Affichage final
            text_widget.insert(END, "\n".join(rapport))
            text_widget.configure(state=DISABLED)

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'affichage des résultats:\n{str(e)}")

    def _afficher_arbre_pi_prime(self, points_array, pi_prime, alpha, beta):
        fenetre = Toplevel(self.parent)
        fenetre.title(f"Arbre π' (α={alpha}, β={beta})")
    
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    
    # Points (mêmes positions que π)
        ax.scatter(points_array[:, 0], points_array[:, 1], c='gray', s=60)
    
    # Arêtes
        for child, parent in pi_prime.items():
            start = points_array[parent-1]
            end = points_array[child-1]
            ax.plot([start[0], end[0]], [start[1], end[1]], 'k-', alpha=0.6)
        
        # Distance sur l'arête
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, f"{self.distances_pi_prime[child]:.2f}", 
                fontsize=9, color='blue')
    
    # Numérotation (1 à n)
        for i in range(len(points_array)):
            ax.scatter(points_array[i][0], points_array[i][1], c='red', s=80)
            ax.text(points_array[i][0]+0.03, points_array[i][1]+0.03, 
                str(i+1), fontsize=12, color='red')
    
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=fenetre)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=1)
    
    # Convertir les points en objets Point
        #points_objects = [Point(p[5], p[6], i) for i, p in enumerate(self.points)]
    # Calculer le clustering
        #Clustering.clustering_hierarchique(points_objects)

    def calculer_facteur_optimal(self): 
        """Calcule le facteur optimal entre π et π' en utilisant les derniers coûts calculés"""
        try:
            # Vérifier que les coûts ont été calculés
            # Vérification des données nécessaires
            required_attrs = ['_last_pi_result', '_last_pi_prime_result']
            missing = [attr for attr in required_attrs if not hasattr(self, attr)]
            
            if missing:
                missing_str = " et ".join(missing)
                messagebox.showerror("Erreur", "Vous devez d'abord :\n"
                                      "1. Créer l'arbre π et calculer son coût\n"
                                      "2. Créer l'arbre π' et calculer son coût\n"
                                      "avec le même nombre de clusters (k)")
                return
            
             # Vérifier que le même k a été utilisé
            if self._last_pi_result['k'] != self._last_pi_prime_result['k']:
                messagebox.showerror("Erreur", 
                            f"Les nombres de clusters ne correspondent pas:\n"
                            f"- Arbre π: k={self._last_pi_result['k']}\n"
                            f"- Arbre π': k={self._last_pi_prime_result['k']}\n\n"
                            "Veuillez recalculer avec le même k.")
                return
        
            k = self._last_pi_result['k']
            cost_pi = self._last_pi_result['cost']
            cost_pi_prime = self._last_pi_prime_result['cost']

            #calculer le cout optimal 
            points_array = np.array([[p[5], p[6]] for p in self.points])
            points_optimal = {i: list(coord) for i, coord in enumerate(points_array)}

            # Vérifier que k_entry existe et contient une valeur valide
            if not hasattr(self, 'k_entry') or not self.k_entry.get().isdigit():
                messagebox.showerror("Erreur", "Veuillez spécifier un nombre valide de clusters")
                return
            
            k_value = int(self.k_entry.get())
            clusters, centers, cost, farthest_points = self.k_center_optimal(points_optimal, k_value)
            self.CoûtOptimal = round(cost, 3)

            if cost_pi_prime == 0:
                messagebox.showerror("Erreur", "Le coût de π' est nul (division impossible)")
                return
            
            # Calcul du facteur
            facteur = cost_pi_prime/ self.CoûtOptimal 
            improvement = (1 - (self.CoûtOptimal/cost_pi_prime)) * 100  # Pourcentage d'amélioration

            # Affichage des résultats
            self.text_resultats.config(state=NORMAL)
            self.text_resultats.delete(1.0, END)

            results = [
                "=== FACTEUR D'OPTIMALITÉ ===",
                f"\nPour k = {k} clusters:",
                f"\n[ARBRE π]",
                f"- Coût (diamètre max): {cost_pi:.4f}",
                f"- Points: {self._last_pi_result['points']}",
                f"- Cluster: {self._last_pi_result['cluster']}",
            
                f"\n[ARBRE π']",
                f"- Coût (diamètre max): {cost_pi_prime:.4f}",
                f"- Points: {self._last_pi_prime_result['points']}",
                f"- Cluster: {self._last_pi_prime_result['cluster']}",

                f"\n[ARBRE π']",
                f"- Coût meilleur clustering k = {k} clusters: : {self.CoûtOptimal:.4f}",


                f"\n[COMPARAISON]",
                f"- Facteur (π'/CoûtOptimal): {facteur:.4f}",
                f"- Amélioration: {improvement:.2f}%",
                "\n[INTERPRÉTATION]",
                #f"- L'arbre π' est {facteur:.2f}x plus efficace" if facteur > 1 
               # else f"- La solution optimale est {1/facteur:.2f}x plus efficace"
            ]
        
            self.text_resultats.insert(END, "\n".join(results))
            self.text_resultats.config(state=DISABLED)
            self.text_resultats.see(END)  # Faire défiler vers le bas
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Calcul du facteur échoué:\n{str(e)}")



    #Affiche l'arbre hiérarchique avec numérotation à partir de 1
    def afficher_arbre_hierarchique(self, Z, k):
        
    # Créer une nouvelle fenêtre
        fenetre_arbre = Toplevel(self.parent)
        fenetre_arbre.title(f"Clustreing Hiérarchique (k={k})")
        fenetre_arbre.geometry("600x400")
    # Zone de texte avec scrollbar
        frame_texte = Frame(fenetre_arbre)
        frame_texte.pack(fill=BOTH, expand=True)
    
        scrollbar = Scrollbar(frame_texte)
        text_widget = Text(frame_texte, height=10, yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_widget.yview)

        scrollbar.pack(side=RIGHT, fill=Y)
        text_widget.pack(side=LEFT, fill=BOTH, expand=True)
    
        # pour calculer des cluster
        clusters = fcluster(Z, k,criterion='maxclust')
        #unique_clusters = np.unique(clusters)
    
       
        text_widget.insert(END, f"Cluster 1:\n")
        points_in_cluster1 = [i for i, cl in enumerate(clusters) if cl == 1]
        for point_idx in points_in_cluster1:
            point_data = self.points[point_idx]
            text_widget.insert(END, f"- Point {point_idx + 1} ({point_data[4]}): "
                                  f"({point_data[5]:.2f}, {point_data[6]:.2f})\n")

        text_widget.insert(END, f"Cluster 2:\n")
        points_in_cluster2 = [i for i, cl in enumerate(clusters) if cl == 2]
        for point_idx in points_in_cluster2:
            point_data = self.points[point_idx]
            text_widget.insert(END, f"- Point {point_idx + 1} ({point_data[4]}): "
                                  f"({point_data[5]:.2f}, {point_data[6]:.2f})\n")

        # Ajouter les distances entre clusters si nécessaire
        text_widget.insert(END, "\nDistances entre clusters:\n")
        for i in range(len(Z)):
            node1, node2, dist, _ = Z[i]
            text_widget.insert(END, f"Nœud {int(node1)+1} + Nœud {int(node2)+2} -> "
                              f"Distance: {dist:.2f}\n")
            

            # Graphique en haut
        fig_frame = Frame(frame_texte)
        fig_frame.pack(side=TOP, fill=BOTH, expand=True)

        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
    
        points_array = np.array([[p[5], p[6]] for p in self.points])
        scatter = ax.scatter(points_array[:,0], points_array[:,1], c=clusters, cmap='tab20')
    
        for i, point in enumerate(self.points):
            ax.text(point[5]+0.05, point[6]+0.05, f"{i+1}", fontsize=12)

        ax.set_title(f"Visualisation des {k} Clusters")
        canvas = FigureCanvasTkAgg(fig, master=fig_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

