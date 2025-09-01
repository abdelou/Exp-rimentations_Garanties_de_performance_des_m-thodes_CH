import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# Import matplotlib en premier
import matplotlib
matplotlib.use('TkAgg')  # Configuration du backend

# Pu's les autres imports
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import simpledialog
import time
from scipy.cluster.hierarchy import linkage, fcluster
import tkinter.messagebox as messagebox
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random  # Nouvel import pour la génération aléatoire
# Vos imports personnels
from euclidienne_distance import Euclidienne_Distance
from espace_metrique import Espace_metrique
from clustering_optimal import Clustering_Optimal
from choix_random import Choix_random 
from point import Point
import inspect
import math
import itertools 
class ChoixConstruction(Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Choix de construction")
        # Configuration de la taille et position centrale
        largeur_fenetre = 700
        hauteur_fenetre = 550  # J'ai augmenté légèrement la hauteur pour plus d'espace
        largeur_ecran = self.winfo_screenwidth()
        hauteur_ecran = self.winfo_screenheight()
    
        pos_x = (largeur_ecran // 2) - (largeur_fenetre // 2)
        pos_y = (hauteur_ecran // 2) - (hauteur_fenetre // 2)
    
        self.geometry(f"{largeur_fenetre}x{hauteur_fenetre}+{pos_x}+{pos_y}")
        self.resizable(False, False)
    
        # Style amélioré
        self.configure(bg="#f0f8ff")
        style = ttk.Style()
    
        # Configuration des styles
        style.configure('TButton', 
                  font=('Arial', 12, 'bold'),
                  padding=10,
                  foreground='navy',
                  background='#e1f5fe')
    
        style.map('TButton',
                background=[('active', '#b3e5fc')],
                foreground=[('active', 'navy')])
    
        # Style spécial pour le bouton Quitter
        style.configure('Quit.TButton', 
                    foreground='white',
                    background='#d32f2f')
    
        # Frame principal avec ombre visuelle
        main_frame = Frame(self, 
                        bg="#f0f8ff",
                        padx=30, 
                        pady=30)
        main_frame.pack(expand=True, fill=tk.BOTH)
    
        # Conteneur pour le contenu principal
        content_frame = Frame(main_frame, bg="#f0f8ff")
        content_frame.pack(expand=True, fill=tk.BOTH)
    
        # Titre stylisé
        Label(content_frame, 
            text="Choisissez le mode de sélection des points",
            font=('Arial', 16, 'bold'),
            bg="#f0f8ff",
            fg="navy").pack(pady=(0, 30))
    
        # Frame pour les boutons principaux
        btn_frame = Frame(content_frame, bg="#f0f8ff")
        btn_frame.pack(expand=True, padx=50)
    
        # Boutons principaux
        buttons = [
            ("Choix manuel des points", self.choix_manuel),
            ("Choix aléatoire des points", self.choix_aleatoire),
            ("Expériences aléatoires", self.executer_experiences),
            ("Expériences sous forme géométriques", self.forme_geometriques)
        ]
    
        for text, command in buttons:
            btn = ttk.Button(btn_frame,
                        text=text,
                        command=command,
                        style='TButton')
            btn.pack(fill=tk.X, pady=8, ipady=8)
    
        # Frame pour le bouton Quitter (en bas de la fenêtre principale)
        quit_frame = Frame(main_frame, bg="#f0f8ff")
        quit_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(0, 10))
    
        # Bouton Quitter centré
        btn_quitter = ttk.Button(quit_frame,
                            text="Quitter",
                            command=self.quitter_application,
                            style='Quit.TButton')
        btn_quitter.pack(side=tk.RIGHT, padx=20, ipady=5, ipadx=20)
    
        # Ajout d'une icône (optionnel)
        try:
            self.iconbitmap('icon.ico')
        except:
            pass
        
    def executer_experiences(self):
        """Ouvre l'interface des expériences aléatoires"""
        try:
            fenetre_exp = tk.Toplevel(self.parent)
            fenetre_exp.title("Expériences Aléatoires")
        
            # Import différé
            from experiences import ExperienceAleatoire
            ExperienceAleatoire(fenetre_exp)
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")
    
    def choix_manuel(self):
        """Affiche l'interface principale de construction"""
        self.destroy()  # Ferme la fenêtre de choix
        self.parent.deiconify()

        #if hasattr(self.parent, 'cadre_principal') and self.parent.cadre_principal:
            #self.parent.cadre_principal.destroy()

        self.parent.setup_main_interface()
    
    def choix_aleatoire(self):
        """Affiche une fenêtre pour la génération aléatoire"""
        self.destroy()
        #self.parent.deiconify()
        #FenetreAleatoire(self.parent)
        """Affiche une fenêtre pour la génération aléatoire"""
        self.destroy()  # Ferme la fenêtre de choix
        # Crée une nouvelle fenêtre pour le mode aléatoire
        fenetre_aleatoire = Toplevel(self.parent)
        fenetre_aleatoire.title("Génération des points aléatoire ")
        fenetre_aleatoire.geometry("1000x800")
        # Utilise votre classe Choix_random
        Choix_random(fenetre_aleatoire)

    def forme_geometriques(self):
        self.destroy()  # Ferme la fenêtre de choix
        self.parent.deiconify()

        #if hasattr(self.parent, 'cadre_principal') and self.parent.cadre_principal:
            #self.parent.cadre_principal.destroy()

        self.parent.interface_forme_geometriques()

    def quitter_application(self):
        """Ferme proprement l'application"""
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter l'application ?"):
            self.parent.destroy()  # Ferme la fenêtre principale
            # Si vous avez d'autres processus à arrêter, ajoutez-les ici

class MainApplication(Tk):
    def __init__(self):
        super().__init__()
        self.title("Construction d'arbres T^π et T^π'")
        self.geometry("1650x1150")
        
        # Cacher la fenêtre principale au départ
        #self.withdraw()
        self.deiconify()

        # Initialiser les attributs
        self.cadre_principal = None
        self.cadre_controles = None
        self.espace = None
        #self.entree_max = None
        #self.frame_espace = None
       # self.espace = None
        #self.espace_visible = False
        self.choix_window = ChoixConstruction(self) 
        self.withdraw()
        # Configurer l'interface principale
        #self.setup_main_interface()
        
        # Démarrer avec la fenêtre de choix
        #self.choix_window = ChoixConstruction(self)  
        #self.creer_fenetre_choix()

    def creer_fenetre_choix(self):
        """Crée et affiche la fenêtre de choix principale"""
        self.choix_window = ChoixConstruction(self)

    def show_choice_window(self):
        """Affiche la fenêtre de choix principale"""
        if hasattr(self, 'choix_window') and self.choix_window:
            self.choix_window.destroy()
        self.choix_window = ChoixConstruction(self)
    
    def setup_main_interface(self, espace=None):
        """Affiche l'interface principale"""
        #self.deiconify()
        if hasattr(self, 'cadre_principal') and self.cadre_principal :
        # Cadre principal
            self.cadre_principal.destroy()
        # Recréer l'interface

        self.cadre_principal = Frame(self)
        self.cadre_principal.pack(expand=True, fill=BOTH)

        # Cadre pour les contrôles
        self.cadre_controles = Frame(self.cadre_principal)
        self.cadre_controles.pack(pady=15)

        # Titre
        self.title_label=Label(self.cadre_principal, 
            text="Construction des arbres T^π et T^π'",
            font=('Arial', 16, 'bold'))
        self.title_label.pack()

        # Widgets
        Label(self.cadre_controles, text="Nombre de points (3-10):").grid(row=0, column=0)
        self.entree_max = Entry(self.cadre_controles, width=5)
        self.entree_max.grid(row=0, column=1, padx=5)
        self.entree_max.insert(0, "")

        # Champ pour la taille axe X
        Label(self.cadre_controles, text="Taille axe X:").grid(row=0, column=2, padx=5)
        self.entree_axe_x = Entry(self.cadre_controles, width=5)
        self.entree_axe_x.grid(row=0, column=3, padx=5)
        self.entree_axe_x.insert(0, "10")

        # Champ pour la taille axe Y
        Label(self.cadre_controles, text="Taille axe Y:").grid(row=0, column=4, padx=5)
        self.entree_axe_y = Entry(self.cadre_controles, width=5)
        self.entree_axe_y.grid(row=0, column=5, padx=5)
        self.entree_axe_y.insert(0, "10")
        # Boutons
        ttk.Button(self.cadre_controles,
                text="afficher espace",
                command=self.afficher_espace).grid(row=0, column=6, padx=5)
        
        ttk.Button(self.cadre_controles,
                text="Initialiser l'espace",
                command=self.to_initialiser_espace).grid(row=0, column=7, padx=5)
        
        ttk.Button(self.cadre_controles,
                text="Supprimer point",
                command=self.supprimer_point).grid(row=0, column=8, padx=5)

        ttk.Button(self.cadre_controles,
                text="Créer arbre π",
                command=self.creer_arbre_pi).grid(row=0, column=9, padx=5)

        ttk.Button(self.cadre_controles,
                text="Créer arbre π'",
                command=self.creer_arbre_pi_prime).grid(row=0, column=10, padx=5)

        ttk.Button(self.cadre_controles,
                text="Facteur Optimal",
                command=self.calculer_facteur_optimal).grid(row=0, column=11, padx=5)
       
        # Ajout du nouveau bouton Quitter
        ttk.Button(self.cadre_controles,
            text="Quitter",
            command=self.retour_choix).grid(row=0, column=12, padx=5)
            
            # Espace métrique
        self.frame_espace = Frame(self.cadre_principal)
        self.frame_espace.pack(expand=True, fill=BOTH, pady=20)
        
            # Utiliser l'espace existant ou en créer un nouveau
        if espace:
            self.espace = espace
            self.espace.parent = self.frame_espace
            self.espace.canvas.pack()
        else:
            self.espace = Espace_metrique(self.frame_espace)
    
        #self.espace_visible = True
    def interface_forme_geometriques(self,espace=None):

        if hasattr(self, 'cadre_principal') and self.cadre_principal:
            # Cadre principal
            self.cadre_principal.destroy()
            # Recréer l'interface

        #self.cadre_principal = Frame(self)
        #self.cadre_principal.pack(expand=True, fill=BOTH)

        # interface  pour les contrôles
        self.cadre_controles = Frame(self.cadre_principal)
        self.cadre_controles.pack(pady=15, fill=tk.X)  # Ajout de fill=tk.X

        # ComboBox pour le choix de la forme géométrique
        Label(self.cadre_controles, text="Forme géométrique:").grid(row=0, column=0, padx=5)
        self.choix_forme = ttk.Combobox(self.cadre_controles, 
                              values=["Diagonale droite", "Diagonale gauche", 
                                       "Droite horizontale", "Droite Verticale","Cercle plein","Carré plein"])
        self.choix_forme.grid(row=0, column=1, padx=5)
        self.choix_forme.set("Choisir une forme")


        Label(self.cadre_controles, text="Nombre de points:").grid(row=0, column=2, padx=5)
        self.nombre_point = ttk.Combobox(self.cadre_controles, 
                              values=[1,2,3,4,5,6,7,8,9,10])
        self.nombre_point.grid(row=0, column=3, padx=5)
        self.nombre_point.set("10")

        # Champ pour la taille axe X
        Label(self.cadre_controles, text="Taille axe X:").grid(row=0, column=4, padx=5)
        self.entree_axe_x = Entry(self.cadre_controles, width=5)
        self.entree_axe_x.grid(row=0, column=5, padx=5)
        self.entree_axe_x.insert(0, "10")

        # Champ pour la taille axe Y
        Label(self.cadre_controles, text="Taille axe Y:").grid(row=0, column=6, padx=5)
        self.entree_axe_y = Entry(self.cadre_controles, width=5)
        self.entree_axe_y.grid(row=0, column=7, padx=5)
        self.entree_axe_y.insert(0, "10")

        # Nouvelle frame pour les boutons en bas
        btn_frame = Frame(self.cadre_principal)
        btn_frame.pack(fill=tk.X, pady=10)


        
        # Bouton Quitter
        button_width = len("Créer la forme géométrique") + 2  # +2 pour un peu de marge
        ttk.Button(self.cadre_controles,
                  text="Créer la forme géométrique",
                  command=self.creer_forme_geometrique,
                  width=button_width).grid(row=4, column=0, padx=15, pady=5, sticky='ew')

        ttk.Button(self.cadre_controles,
                  text="Calculer Facteur",
                  command=self.calcule_facteur,
                  width=button_width).grid(row=5, column=0, padx=15, pady=5, sticky='ew')
        ttk.Button(self.cadre_controles,
                text="Quitter",
                command=self.retour_choix,
                width=button_width).grid(row=6, column=0, padx=15, pady=5, sticky='ew')
        # Espace métrique pour afficher les formes
        self.frame_espace = Frame(self.cadre_principal)
        self.frame_espace.pack(expand=True, fill=BOTH, pady=20)

    def afficher_espace_metrique(self, points, taille_axe_x, taille_axe_y):

        """Affiche les points dans un espace métrique avec axes X et Y"""
        # Créer un canvas si il n'existe pas déjà
        if not hasattr(self, 'canvas_metrique'):
            self.canvas_metrique = tk.Canvas(self.frame_espace, bg='white', 
                                        width=800, height=600)
            self.canvas_metrique.pack(expand=True, fill=tk.BOTH)
        else:
         # Effacer le canvas existant
            self.canvas_metrique.delete('all')
    
        # Calculer les facteurs d'échelle
        canvas_width = self.canvas_metrique.winfo_width()
        canvas_height = self.canvas_metrique.winfo_height()
    
        scale_x = (canvas_width - 100) / taille_axe_x  # Marge de 50 pixels de chaque côté
        scale_y = (canvas_height - 100) / taille_axe_y
    
        # Dessiner les axes X et Y
        offset = 50  # Marge pour les axes
    
        # Axe X
        self.canvas_metrique.create_line(offset, canvas_height - offset, 
                                    canvas_width - offset, canvas_height - offset, 
                                    width=2, arrow=tk.LAST)
        # Axe Y
        self.canvas_metrique.create_line(offset, canvas_height - offset, 
                                    offset, offset, 
                                    width=2, arrow=tk.LAST)
    
        # Ajouter les étiquettes des axes
        self.canvas_metrique.create_text(canvas_width - offset + 20, canvas_height - offset, 
                                   text="X", font=('Arial', 12, 'bold'))
        self.canvas_metrique.create_text(offset, offset - 20, 
                                   text="Y", font=('Arial', 12, 'bold'))
    
        # Ajouter les graduations
        # Graduations axe X
        for i in range(0, int(taille_axe_x) + 1):
            x_pos = offset + i * scale_x
            self.canvas_metrique.create_line(x_pos, canvas_height - offset - 5, 
                                       x_pos, canvas_height - offset + 5, 
                                       width=2)
            self.canvas_metrique.create_text(x_pos, canvas_height - offset + 20, 
                                       text=str(i), font=('Arial', 10))
    
        # Graduations axe Y
        for i in range(0, int(taille_axe_y) + 1):
            y_pos = canvas_height - offset - i * scale_y
            self.canvas_metrique.create_line(offset - 5, y_pos, 
                                       offset + 5, y_pos, 
                                       width=2)
            self.canvas_metrique.create_text(offset - 20, y_pos, 
                                       text=str(i), font=('Arial', 10))
    
            # Ajouter le CADRE DU CERCLE si c'est la forme sélectionnée
        if hasattr(self, 'choix_forme') and self.choix_forme.get() == "Cercle plein":
            center_x, center_y = taille_axe_x/2, taille_axe_y/2
            radius = min(taille_axe_x, taille_axe_y) * 0.1
        
            # Convertir en coordonnées canvas
            canvas_center_x = offset + center_x * scale_x
            canvas_center_y = canvas_height - offset - center_y * scale_y
            canvas_radius = radius * scale_x
        
            # Dessiner le cercle (couleur verte, épaisseur 2)
            self.canvas_metrique.create_oval(
                canvas_center_x - canvas_radius, canvas_center_y - canvas_radius,
                canvas_center_x + canvas_radius, canvas_center_y + canvas_radius,
                outline='#2ecc71', width=2, dash=(4,2)  # Vert avec motif pointillé
            )
        # Dessiner les points
        point_size = 5  # Taille des points
        for point in points:
            x, y = point
            # Convertir les coordonnées réelles en coordonnées canvas
            canvas_x = offset + x * scale_x
            canvas_y = canvas_height - offset - y * scale_y
        
            # Dessiner le point
            self.canvas_metrique.create_oval(
                canvas_x - point_size, canvas_y - point_size,
                canvas_x + point_size, canvas_y + point_size,
                fill='red', outline='black'
            )
    
        # Ajouter un titre
        self.canvas_metrique.create_text(canvas_width // 2, 30, 
                                   text="Espace Métrique ({} points)".format(len(points)),
                                   font=('Arial', 14, 'bold'))
    
        print("Espace métrique affiché avec {} points".format(len(points)))

    def creer_forme_geometrique(self):
        """Crée et affiche une forme géométrique dans l'espace métrique"""
        try:
            # Récupérer les paramètres
            forme = self.choix_forme.get()
            if forme == "Choisir une forme":
                messagebox.showerror("Erreur", "Veuillez sélectionner une forme géométrique")
                return

            
            nb_points = int(self.nombre_point.get())
            taille_x = float(self.entree_axe_x.get())
            taille_y = float(self.entree_axe_y.get())
        
            # Vérifier les valeurs
            if nb_points <= 0:
                messagebox.showerror("Erreur", "Le nombre de points doit être positif")
                return
            if taille_x <= 0 or taille_y <= 0:
                messagebox.showerror("Erreur", "Les tailles d'axes doivent être positives")
                return 
    
            # Générer les points selon la forme
            points = self.generer_points_aleatoires_selon_forme(forme, nb_points, taille_x, taille_y)
            
            if points is None:
                return


            #Ajoute les points dans l'espace
            self.afficher_espace_metrique(points,taille_x,taille_y)

        except ValueError as e:
            messagebox.showerror("Erreur", f"Veuillez entrer des valeurs numériques valides: {str(e)}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")

        #cette fonctionner  retourner les points et chaque points dans quelle position 
        # dans l'espace metrique 
    def generer_points_aleatoires_selon_forme(self, forme, nb_points, taille_x, taille_y):
            # Créer les points selon la forme géométrique choisie
        points = []
        margin = 0.1  # Réduire la marge pour mieux remplir l'espace



        if forme == "Diagonale droite":
            # Points sur la diagonale droite avec espacement aléatoire
            for _ in range(nb_points):
                x = random.uniform(margin*taille_x, (1-margin)*taille_x)
                y = x * (taille_y/taille_x)  # Maintient la relation y = x
                points.append([x, y])

        elif forme == "Diagonale gauche":
            # Points sur la diagonale gauche avec espacement aléatoire
            for _ in range(nb_points):
                x = random.uniform(margin*taille_x, (1-margin)*taille_x)
                y = taille_y - x * (taille_y/taille_x)  # Maintient la relation y = -x
                points.append([x, y])
        
        elif forme == "Droite horizontale":
            # Points sur une ligne horizontale avec espacement aléatoire
            y = taille_y / 2
            for _ in range(nb_points):
                x = random.uniform(margin*taille_x, (1-margin)*taille_x)
                points.append([x, y])
        
        elif forme == "Droite Verticale":
            # Points sur une ligne verticale avec espacement aléatoire
            x = taille_x / 2
            for _ in range(nb_points):
                y = random.uniform(margin*taille_y, (1-margin)*taille_y)
                points.append([x, y])
        

        
        elif forme == "Cercle plein":
            # Centre du cercle
            center_x, center_y = taille_x/2, taille_y/2
            # Rayon à 40% de la plus petite dimension pour garder une marge
            radius = min(taille_x, taille_y) * 0.1    
            # Méthode optimisée pour générer des points aléatoires dans le cercle
            for _ in range(nb_points):
                # 1. Générer un angle aléatoire
                angle = random.uniform(0, 2*math.pi)
                # 2. Générer un rayon aléatoire (racine carrée pour distribution uniforme)
                r = radius * math.sqrt(random.random())
        
                # 3. Calculer les coordonnées
                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)
                points.append([x, y])
    
            # Tri par angle pour un meilleur rendu visuel
            points = sorted(points, key=lambda p: math.atan2(p[1]-center_y, p[0]-center_x))
        
        elif forme == "Carré plein":
                # Paramètres du carré avec 10% de marge
            margin = 0.45
            cote = min(taille_x, taille_y) * (1 - 2*margin)
            start_x = (taille_x - cote)/2
            start_y = (taille_y - cote)/2
    
            # Générer 2 points par côté (8 points au total)
            for i in range(2):
                # Côté bas (y = start_y)
                x = random.uniform(start_x, start_x + cote)
                points.append([x, start_y])
        
                # Côté haut (y = start_y + cote)
                x = random.uniform(start_x, start_x + cote)
                points.append([x, start_y + cote])
        
                # Côté gauche (x = start_x)
                y = random.uniform(start_x, start_x + cote)
                points.append([start_x, y])
        
                # Côté droit (x = start_x + cote)
                y = random.uniform(start_x, start_x + cote)
                points.append([start_x + cote, y])
    
            # Pour les points restants (nb_points - 8), les placer à l'intérieur du carré
            for _ in range(nb_points - 8):
                x = random.uniform(start_x + 0.1*cote, start_x + 0.9*cote)
                y = random.uniform(start_y + 0.1*cote, start_y + 0.9*cote)
                points.append([x, y])
    
            # Si nb_points < 8, ne garder que les premiers points
            points = points[:nb_points]
            
        else:
            messagebox.showerror("Erreur", "Forme géométrique invalide")
            return None
    
        return np.array(points)
    
    def meilleur_clustreing_optimal_aleatoir(self,points,k_cluster):

        points_optimal = {i: list(coord) for i, coord in enumerate(points)}
        # === 3. Fonction de coût (rayon max pour un clustering donné)
        #dist_matrix = distance_matrix(coords, coords)
        def k_center_optimal(points_dict, k_cluster):
            # Préparation des données à partir du dictionnaire
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
        print("cout optimal",CoûtOptimal)
        return CoûtOptimal,points
        

    def calcule_facteur(self,):
            
        """Calcule le facteur entre le coût de π' et le coût optimal"""
        try:
            # Récupérer les paramètres depuis l'interface
            nb_points = int(self.nombre_point.get())
            axe_x = float(self.entree_axe_x.get())
            axe_y = float(self.entree_axe_y.get())

            if nb_points < 3 :  
                messagebox.showerror("Erreur", "Le nombre de points doit être au moins 3")

                return
            
            nb_clusters = simpledialog.askinteger("Nombre de clusters", 
                                            "Entrez le nombre de clusters:",
                                            parent=self,
                                            minvalue=2, 
                                            maxvalue=nb_points-1)

            

            if nb_clusters is None:  # Si l'utilisateur annule
                return
            forme=self.choix_forme.get()
            # Calculer le coût optimal
            points_forme=self.generer_points_aleatoires_selon_forme(forme,nb_points,axe_x,axe_y)
            cout_optimal,points=self.meilleur_clustreing_optimal_aleatoir(points_forme,nb_clusters)
        
            # Calculer les arbres π et π'
            D = distance_matrix(points_forme, points_forme)
            selected_indices, R_values, parent_indices = Choix_random.farthest_first_tree(points_forme)
            pi_prime, edge_lengths_pi, _, _ = Choix_random.hierarchical_tree(
                D, selected_indices, R_values)
        
            # Calculer les coûts
            resultats_df = Choix_random.comparer_k_clustering_evolutif(
                points_forme, selected_indices, parent_indices, 
                pi_prime, edge_lengths_pi, nb_clusters)
        
            # Extraire les résultats
            cout_ff = resultats_df.loc[
                resultats_df["Méthode"] == "Farthest-First Traversal", 
                "Coût final"].values[0]
            cout_pi_prime = resultats_df.loc[
                resultats_df["Méthode"] == "Arbre hiérarchique T^π′", 
                "Coût final"].values[0]
        
            # Calculer le facteur
            facteur = cout_pi_prime / cout_optimal if cout_optimal != 0 else float('inf')
        

            # Afficher les résultats
            message = (
                f"Résultats pour {nb_points} points et {nb_clusters} clusters:\n\n"
                f"Coût FF (Farthest-First): {cout_ff:.2f}\n"
                f"Coût π': {cout_pi_prime:.2f}\n"
                f"Coût Optimal: {cout_optimal:.2f}\n"
                f"Facteur (π'/Optimal): {facteur:.2f}"
            )
            messagebox.showinfo("Résultats du calcul", message)
            
        except ValueError as e:
            messagebox.showerror("Erreur", f"Valeur incorrecte: {str(e)}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")
        
    def show_main_interface(self, espace=None):
        # Utiliser l'espace existant ou en créer un nouveau
        #self.setup_main_interface()  # Crée l'interface si ce n'est pas déjà fait
        self.cadre_controles.pack_forget()
        if self.cadre_principal is None:
            self.setup_main_interface()

        if espace:
            if hasattr(self,'espace') and self.espace:
                self.espace.canvas.pack_forget()
            
            self.espace = espace
            self.espace.parent = self.frame_espace
            self.espace.canvas.pack()

        mode = "Aléatoire" if espace else "Manuel"
        self.title_label.config(text=f"Construction des arbres T^π et T^π' - Mode {mode}")    

    def to_initialiser_espace(self):
        try:
            n = int(self.entree_max.get())
            if not 3 <= n <= 10:
                messagebox.showerror("Erreur", "Le nombre de points doit être entre 3 et 10")
                return
            self.espace.reinitialiser_espace(n)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer un nombre entre 3 et 10")
    def afficher_espace(self,coords=None):
        try:
            # Récupérer les valeurs des axes
            taille_x = float(self.entree_axe_x.get())
            taille_y = float(self.entree_axe_y.get())
        
            # Vérifier que les valeurs sont valides
            if taille_x <= 0 or taille_y <= 0:
                messagebox.showerror("Erreur", "Les tailles d'axes doivent être positives")
                return
            
            # Si l'espace n'existe pas encore, le créer
            if not hasattr(self, 'espace') or self.espace is None:
                self.espace = Espace_metrique(self.frame_espace, taille_x, taille_y)
            else:

                # Sinon, mettre à jour les dimensions de l'espace existant
                if hasattr(self.espace, 'set_dimensions'):
                    self.espace.set_dimensions(taille_x, taille_y)
                else:
                    # Si la méthode set_dimensions n'existe pas, recréer l'espace
                    self.espace.canvas.destroy()
                    self.espace = Espace_metrique(self.frame_espace, taille_x, taille_y)
                
            # S'assurer que l'espace est bien affiché
            self.espace.canvas.pack(expand=True, fill=BOTH)
            if coords is not None:
                self.espace.positions_pi = np.array(coords)
             # Si besoin, réinitialiser selected_indices, R_values, etc.
                self.espace.selected_indices = list(range(len(coords)))
            # Recalcul ou réinitialisation éventuelle de R_values ici selon ta logique
                self.espace.R_values = self.espace.calculer_R_values()  # méthode hypothétique à créer
        
                self.espace.canvas.pack(expand=True, fill=BOTH)

                # Appeler la création de l'arbre π'
            self.espace.creer_arbre_pi_prime(alpha=1, beta=2)
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques valides pour les tailles d'axes")
    
    def supprimer_point(self):
        self.espace.retirer_point_selectionne()
    
    def creer_arbre_pi(self):
        self.espace.creer_arbre_pi()
    
    def creer_arbre_pi_prime(self):
        self.espace.creer_arbre_pi_prime()
    
    def calculer_facteur_optimal(self):
        self.espace.calculer_facteur_optimal()

    def retour_choix(self):
        """Retourne à la fenêtre de choix principale"""
        self.withdraw()  # Cache la fenêtre principale
        self.creer_fenetre_choix()  # Recrée la fenêtre de choix

    def choix_aleatoire(self):
        """Affiche une fenêtre pour la génération aléatoire"""
        self.destroy()
        #self.parent.deiconify()  # Affiche la fenêtre principale
        # Créer une nouvelle fenêtre Toplevel pour le mode aléatoire
        fenetre_aleatoire = Toplevel(self.parent)
        #fenetre_aleatoire.title("Génération aléatoire de points")
        #fenetre_aleatoire.geometry("800x600")

        Choix_random(fenetre_aleatoire)  # Utilisez votre classe Choix_random

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()