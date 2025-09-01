import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'
import csv  
import matplotlib
matplotlib.use('TkAgg')  
import tkinter as tk
from tkinter import *
from tkinter import ttk,messagebox
from tkinter import simpledialog
import time
from scipy.cluster.hierarchy import linkage, fcluster
import tkinter.messagebox as messagebox
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random  
from euclidienne_distance import Euclidienne_Distance
from espace_metrique import Espace_metrique
from choix_random import Choix_random 
from clustering_optimal import Clustering_Optimal
from point import Point
from main import MainApplication
from tkinter import Toplevel, Frame, BOTH
from matplotlib.figure import Figure
import networkx as nx
import traceback

class ExperienceAleatoire:
        
    def __init__(self, parent):

        self.parent = parent
        self.setup_ui()

    def setup_ui(self):
       
        self.parent.geometry("500x300")
        main_frame = ttk.Frame(self.parent, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.arbres_frame = ttk.Frame(self.parent)
        self.arbres_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
        Label(main_frame, 
              text="Paramètres des Expériences",
              font=('Arial', 14)).grid(row=0, columnspan=2, pady=10)
        


        ttk.Label(main_frame, text="Nombre d'expériences:").grid(row=1, column=0, sticky='e')
        self.combo_exp = ttk.Combobox(main_frame, 
                             values=[1,10, 20, 30, 40, 50, 60, 80, 100, 200, 300, 
                                    400, 500, 600, 700, 800,1000,2000,3000,4000,5000,30000],
                             state='readonly')  
        self.combo_exp.grid(row=1, column=1, pady=5)
        self.combo_exp.set(10)  
        #ttk.Label(main_frame, text="Nombre d'expériences:").grid(row=1, column=0, sticky='e')
       # self.entry_exp = ttk.Entry(main_frame)
        #self.entry_exp.grid(row=1, column=1, pady=5)
        
        #ttk.Label(main_frame, text="Nombre de points:").grid(row=2, column=0, sticky='e')
        #self.entry_points = ttk.Entry(main_frame)
        #self.entry_points.grid(row=2, column=1, pady=5)
        
        ttk.Label(main_frame, text="Nombre de points:").grid(row=2, column=0, sticky='e')
        self.combo_nbp = ttk.Combobox(main_frame, 
                             values=[3, 4, 5,  6, 7, 8, 9, 10, 
                                    ],
                             state='readonly')  
        self.combo_nbp.grid(row=2, column=1, pady=5)
        self.combo_nbp.set(3)  

        #ttk.Label(main_frame, text="Nombre de cluster:").grid(row=3, column=0, sticky='e')
        #self.entry_cluster = ttk.Entry(main_frame)
        #self.entry_cluster.grid(row=3, column=1, pady=5)

        ttk.Label(main_frame, text="Nombre de cluster:").grid(row=3, column=0, sticky='e')
        self.combo_nbc = ttk.Combobox(main_frame, 
                             values=[2,3, 4, 5,  6, 7, 8, 9, 10, 
                                    ],
                             state='readonly') 
        self.combo_nbc.grid(row=3, column=1, pady=5)
        self.combo_nbc.set(2)  # Valeur par défaut

        #ttk.Label(main_frame, text="Axe X max:").grid(row=4, column=0, sticky='e')
        #self.entry_x = ttk.Entry(main_frame)
        #self.entry_x.grid(row=4, column=1, pady=5)

        ttk.Label(main_frame, text="Taille Axe X:").grid(row=4, column=0, sticky='e')
        self.combo_axeX = ttk.Combobox(main_frame, 
                             values=[10,20, 30, 40, 50 
                                    ],
                             state='readonly') 
        self.combo_axeX.grid(row=4, column=1, pady=5)
        self.combo_axeX.set(10)  
        #ttk.Label(main_frame, text="Axe Y max:").grid(row=5, column=0, sticky='e')
        #self.entry_y = ttk.Entry(main_frame)
        #self.entry_y.grid(row=5, column=1, pady=5)

        ttk.Label(main_frame, text="Taille Axe X:").grid(row=5, column=0, sticky='e')
        self.combo_axeY = ttk.Combobox(main_frame, 
                             values=[10,20, 30, 40, 50 
                                    ],
                             state='readonly')  
        self.combo_axeY.grid(row=5, column=1, pady=5)
        self.combo_axeY.set(10)  
        
        btn_lancer = ttk.Button(main_frame, 
                              text="Lancer les expériences",
                              command=self.lancer_experiences)
        btn_lancer.grid(row=6, columnspan=2, pady=10)

    def afficher_arbres_depuis_donnees(self):
        
        nb_exp = int(self.combo_exp.get())
        nb_points = int(self.combo_nbp.get())
        nb_clusters = int(self.combo_nbc.get())
        axe_x = int(self.combo_axeX.get())
        axe_y = int(self.combo_axeY.get())
        
        coords_ff, coords_pi,selected_indices, parent_indices,R_values,edge_lengths_pi,pi_prime,D = Choix_random.calculer_facteur(
                        parent=self.parent,
                        nb_experiences=1,
                        nb_points=nb_points,
                        nb_clusters=nb_clusters,
                        axe_x=axe_x,
                        axe_y=axe_y
                    )
        G_ff = nx.Graph()
        for i in range(1, len(selected_indices)):
            child = selected_indices[i]
            parent_idx = parent_indices[i]
            num_child = selected_indices.index(child) + 1
            num_parent = selected_indices.index(parent_idx) + 1
            weight = np.linalg.norm(np.array(points[child]) - np.array(points[parent_idx]))
            G_ff.add_edge(num_child, num_parent, weight=weight)

       
        G_pi = nx.Graph()
        for (child_num, parent_num), weight in edge_lengths_pi.items():
            G_pi.add_edge(child_num, parent_num, weight=weight)

     
        pos_ff = {node: coords_ff[node] for node in G_ff.nodes()}
        pos_pi = {node: coords_pi[node] for node in G_pi.nodes()}

        plt.figure(figsize=(12, 6))

       
        plt.subplot(1, 2, 1)
        nx.draw(G_ff, pos=pos_ff, with_labels=True, node_color='skyblue', edge_color='gray', node_size=300)
        plt.title("Arbre Farthest-First Traversal (π)")

        
        plt.subplot(1, 2, 2)
        nx.draw(G_pi, pos=pos_pi, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=300)
        plt.title("Arbre Hiérarchique (π')")

        plt.show()

    def afficher_partitions(self):
        try:
       
            if not hasattr(self, 'partitions_maximales') or not self.partitions_maximales:
                raise ValueError("Aucun cluster disponible")
            
            if not hasattr(self, 'points_maximaux') or len(self.points_maximaux) == 0:
                raise ValueError("Points non disponibles")
        
      
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = plt.cm.tab10.colors
        
           
            for i, cluster in enumerate(self.partitions_maximales):
                try:
                    points = []
                    for p in cluster:
                        if 0 <= int(p) < len(self.points_maximaux):
                            points.append(self.points_maximaux[int(p)])
                
                    if points:
                        points = np.array(points)
                        ax.scatter(points[:,0], points[:,1], 
                                color=colors[i%10], 
                                label=f'Cluster {i+1} ({len(points)} points)')
                except Exception as e:
                    print(f"Erreur Cluster {i}: {str(e)}")
                    continue
        
            ax.set_title("Clusters Optimaux")
            ax.set_xlabel("Axe X")
            ax.set_ylabel("Axe Y")
            ax.legend()
            ax.grid(True)
            plt.show()
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'afficher : {str(e)}")
            print(f"DEBUG - Clusters: {getattr(self, 'partitions_maximales', None)}")
            print(f"DEBUG - Points: {getattr(self, 'points_maximaux', None)}")
    def lancer_experiences(self):
        try:
         
            nb_exp = int(self.combo_exp.get())
            nb_points = int(self.combo_nbp.get())
            nb_clusters = int(self.combo_nbc.get())
            axe_x = int(self.combo_axeX.get())
            axe_y = int(self.combo_axeY.get())

            self.coord_max_facteur_ff =[]  
            self.coord_max_facteur_pi_prime=[]
            self.coord_max_facteur_cluster=[]
            self.coord_max_facteur_points=[]
            self.coord_max_facteur_selected_indices = []
            self.coord_max_facteur_parent_indices = []
            self.coord_max_facteur_R_values = []
            self.coordonner_optimale=[]
            self.coordonner_cout_diametre=[]
            self.coord_max_facteur_D=[]

            result_window = Toplevel(self.parent)
            result_window.title(f"Résultats des {nb_exp} expériences")
            result_window.geometry("1000x600")
        
         
            main_frame = Frame(result_window)
            main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
           
            table_frame = Frame(main_frame)
            table_frame.pack(fill=BOTH, expand=True)
        
           
            tree = ttk.Treeview(table_frame, columns=("Expérience", "Coût FF", "Coût π","CoûtOptimal","cout_diametre","facteur: π/cout_diametre", "facteur π/CoûtOptimal"), show="headings")
        
         
            tree.heading("Expérience", text="Expérience")
            tree.heading("Coût FF", text="Coût FF")
            tree.heading("Coût π", text="Coût π")
            tree.heading("CoûtOptimal", text="CoûtOptimal")
            tree.heading("cout_diametre", text="cout_diametre")
            tree.heading("facteur: π/cout_diametre", text="facteur: π/cout_diametre")
            tree.heading("facteur π/CoûtOptimal", text="facteur π/CoûtOptimal")
    
            tree.column("Expérience", width=100, anchor='center')
            tree.column("Coût FF", width=150, anchor='center')
            tree.column("Coût π", width=150, anchor='center')
            tree.column("CoûtOptimal", width=150, anchor='center')
            tree.column("cout_diametre", width=150, anchor='center')
            tree.column("facteur: π/cout_diametre", width=150, anchor='center')
            tree.column("facteur π/CoûtOptimal", width=200, anchor='center')
        
         
            vsb = ttk.Scrollbar(table_frame, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
            
            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")
        
           
            table_frame.grid_rowconfigure(0, weight=1)
            table_frame.grid_columnconfigure(0, weight=1)
        
     
            button_frame = Frame(main_frame)
            button_frame.pack(fill=X, pady=10)
        
            
            export_button = ttk.Button(button_frame, text="Exporter vers CSV", 
                                    command=lambda: self.exporter_vers_csv(tree))
            export_button.pack(side=LEFT, padx=5)

            afficher_graphe = ttk.Button(button_frame, text="AfficherLesArbres", 
                                    command=lambda: self.afficher_arbres_depuis_donnees())
            afficher_graphe.pack(side=LEFT, padx=5)

            afficher_partitions = ttk.Button(button_frame, text="Afficher Partitions", 
                                    command=lambda: self.afficher_partitions())
            afficher_partitions.pack(side=LEFT, padx=5)

           
            stats_frame = Frame(main_frame)
            stats_frame.pack(fill=X, pady=10)
        
            
            self.facteur = []
            self.facteur_cout_similaire=[]
            self.facteur_cout_similaire_max_global=float('-inf') 
            self.facteur_max_global = float('-inf')  

            self.stats_label = Label(stats_frame, font=('Arial', 12, 'bold'))
            self.stats_label.pack()
            self.stats_label_diametre = Label(stats_frame, font=('Arial', 12, 'bold'))
            self.stats_label_diametre.pack()
            self.coord_label = Label(stats_frame, font=('Arial', 10))
            self.coord_label.pack()
            self.cluster_label = Label(stats_frame, font=('Arial', 10))
            self.cluster_label.pack()

            def executer_experience(exp_num):
                try:
                    resultats = Choix_random.calculer_facteur(
                        parent=self.parent,
                        nb_experiences=1,
                        nb_points=nb_points,
                        nb_clusters=nb_clusters,
                        axe_x=axe_x,
                        axe_y=axe_y
                    )
                    
                    if not resultats.empty:
                        row = resultats.iloc[0]
                        facteur = row["facteur: π/CoûtOptimal"]
                        facteur_cout_similaire=row["facteur: π/cout_diametre"]
                        self.facteur.append(facteur)
                        self.facteur_cout_similaire.append(facteur_cout_similaire)

                        
                        if facteur > self.facteur_max_global:
                            self.facteur_max_global = facteur
                            

                            self.coord_max_facteur_ff = getattr(self.parent, 'coords_ff', [])
                            self.coord_max_facteur_pi_prime = getattr(self.parent, 'coords_pi', [])
                            self.coord_max_facteur_cluster = getattr(self.parent, 'clusters', [])
                            self.coord_max_facteur_points = np.array(getattr(self.parent, 'points',[]))
                            self.coord_max_facteur_selected_indices = getattr(self.parent, 'selected_indices', [])
                            self.coord_max_facteur_parent_indices = getattr(self.parent, 'parent_indices', [])
                            self.coord_max_facteur_R_values = getattr(self.parent, 'R_values', [])
                            self.coord_max_facteur_D = getattr(self.parent, 'D', [])
                            self.coordonner_optimale = getattr(self.parent, 'points_optimale', [])
                            self.coordonner_cout_diametre=getattr(self.parent,'cout_diametre',[])
                           # self.coordonner_facteur_cout_similaire=getattr(self.parent,'facteur: π/cout_diametre',[])


                           
                            if hasattr(self, 'tree_window') and self.tree_window.winfo_exists():
                                    self.tree_window.destroy()
                            if hasattr(self, 'pi_prime_window') and self.pi_prime_window.winfo_exists():
                                    self.pi_prime_window.destroy()
                                            
                            # Afficher FFT
                            self.afficher_arbre_fft(result_window, facteur)

                            # Afficher π'
                            self.afficher_arbre_pi_prime(result_window, facteur, self.coord_max_facteur_D)
                           
                           #petite affichage pour verifer le calcule 
                            print("\n=== VÉRIFICATION DES DONNÉES ===")
                            print(f"Nombre de points: {len(self.coord_max_facteur_points)}")
                            print(f"Indices sélectionnés: {self.coord_max_facteur_selected_indices}")
                            print(f"Parents: {self.coord_max_facteur_parent_indices}")
                            print(f"Valeurs R: {self.coord_max_facteur_R_values}")  
                         
                            print("=== NOUVEAU MAX TROUVÉ ===")
                            print(f"Points shape: {self.coord_max_facteur_points.shape}")
                            print(f"Selected indices: {self.coord_max_facteur_selected_indices}")
                            print(f"Parent indices: {self.coord_max_facteur_parent_indices}")
                            print(f"R values: {self.coord_max_facteur_R_values}")

                   
                            print("=== NOUVEAU MAX TROUVÉ ===")
                            print(f"Facteur: {facteur:.2f}")
                            print(f"Points: {self.coord_max_facteur_points}")
                            print(f"les Clusters optimal de cette experience avec la facteur_maximal est : {self.coord_max_facteur_cluster},{self.facteur_max_global}")

                            
                            with open("coordonne_points.txt", "w") as f:
                                f.write(f"Coords FF: {self.coord_max_facteur_ff}\n")
                                f.write(f"Coords π: {self.coord_max_facteur_pi_prime}\n")
                                f.write(f"Points: {self.coord_max_facteur_points}\n")
                            
                            with open("partitions_maximales.txt", "w") as f:
                                for i, cluster in enumerate(self.coord_max_facteur_cluster):
                                    f.write(f"Cluster {i+1}: {cluster}\n")

                      
                        tree.insert("", "end", values=(
                            f"Expérience {exp_num}",
                            f"{row['Coût FF']:.2f}",
                            f"{row['Coût π']:.2f}",
                            f"{row['CoûtOptimal']:.2f}",
                            f"{row['cout_diametre']:.2f}",
                            f"{facteur_cout_similaire:.2f}" if not pd.isna(facteur_cout_similaire) else "N/A",
                            f"{facteur:.2f}" if not pd.isna(facteur) else "N/A"
                        ))

                        if self.facteur_cout_similaire:
                            print("Valeur maximale :", max(self.facteur_cout_similaire))
                            self.stats_label_diametre.config(
                                text=f"Statistiques - Max_facteur_coût_similaire: {max(self.facteur_cout_similaire):.2f} | Min: {min(self.facteur_cout_similaire):.2f}"
                            )
                        else:
                            print("La liste est vide.")
                            
                        if self.facteur:
                            self.stats_label.config(
                                text=f"Statistiques - Max_facteur_coût_différent {max(self.facteur):.2f} | Min: {min(self.facteur):.2f}"
                            )
                           
                            self.coord_label.config(
                                text=f"Coordonnées du facteur maximal : {self.coordonner_optimale}"
                            )
                            self.cluster_label.config(
                                text=f"Cluster_optimale : {self.coord_max_facteur_cluster}"
                            
                        )
                        
                        tree.yview_moveto(1)
                
                    if exp_num < nb_exp:
                        result_window.after(100, lambda: executer_experience(exp_num + 1))
                    else:
                         if self.coordonner_optimale and self.coord_max_facteur_cluster:
                            Clustering_Optimal.afficher_clusters(
                                    self.coordonner_optimale,
                                    self.coord_max_facteur_cluster
                        )
                except Exception as e:
                    print(f"Erreur dans l'expérience {exp_num}: {str(e)}")
                    if exp_num < nb_exp:
                        result_window.after(100, lambda: executer_experience(exp_num + 1))

    
            executer_experience(1)
        
        except ValueError:
            messagebox.showerror("Erreur", "Veuillez entrer des nombres valides")
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur est survenue: {str(e)}")

    def afficher_arbre_fft(self, parent_window, facteur):
        """Affiche l'arbre FFT"""
        self.tree_window = Toplevel(parent_window)
        self.tree_window.title(f"Arbre FFT (Facteur max: {facteur:.2f})")
    
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    

        ax.scatter(self.coord_max_facteur_points[:, 0], 
                self.coord_max_facteur_points[:, 1], 
                c='gray', s=60)
    
     
        for i in range(1, len(self.coord_max_facteur_selected_indices)):
            child_idx = self.coord_max_facteur_selected_indices[i]
            parent_idx = self.coord_max_facteur_parent_indices[i]
        
            start = self.coord_max_facteur_points[parent_idx]
            end = self.coord_max_facteur_points[child_idx]
        
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                'k--', linewidth=1)

            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, f"{self.coord_max_facteur_R_values[i]:.2f}", 
                fontsize=9, color='blue')


        for k, idx in enumerate(self.coord_max_facteur_selected_indices):
            ax.scatter(self.coord_max_facteur_points[idx][0], 
                    self.coord_max_facteur_points[idx][1], 
                    c='red', s=80)
            ax.text(self.coord_max_facteur_points[idx][0]+0.03, 
                self.coord_max_facteur_points[idx][1]+0.03,
                str(k+1), fontsize=12, color='red')
    
        ax.grid(True)
        ax.set_aspect('equal')
    
        canvas = FigureCanvasTkAgg(fig, master=self.tree_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    def afficher_arbre_pi_prime(self, parent_window, facteur, D):
       
        self.pi_prime_window = Toplevel(parent_window)
        self.pi_prime_window.title(f"Arbre π' (Facteur max: {facteur:.2f})")
    
      
        pi_prime, edge_lengths, _, _ = Choix_random.hierarchical_tree(
            D, 
            self.coord_max_facteur_selected_indices,
            self.coord_max_facteur_R_values
        )
    
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
    
        ax.scatter(self.coord_max_facteur_points[:, 0], 
                self.coord_max_facteur_points[:, 1], 
                c='gray', s=60)
    
        for (child_num, parent_num), dist in edge_lengths.items():
            child_idx = self.coord_max_facteur_selected_indices[child_num-1]
            parent_idx = self.coord_max_facteur_selected_indices[parent_num-1]
        
            start = self.coord_max_facteur_points[parent_idx]
            end = self.coord_max_facteur_points[child_idx]
        
            ax.plot([start[0], end[0]], [start[1], end[1]], 
                    'b-', linewidth=1)
      
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y, f"{dist:.2f}", 
                fontsize=9, color='darkblue')

   
        for k, idx in enumerate(self.coord_max_facteur_selected_indices):
            ax.scatter(self.coord_max_facteur_points[idx][0], 
                    self.coord_max_facteur_points[idx][1], 
                    c='green', s=80)
            ax.text(self.coord_max_facteur_points[idx][0]+0.03, 
                self.coord_max_facteur_points[idx][1]+0.03,
                str(k+1), fontsize=12, color='green')
    
        ax.grid(True)
        ax.set_aspect('equal')
    
        canvas = FigureCanvasTkAgg(fig, master=self.pi_prime_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=BOTH, expand=True)

    
    
    def afficher_graphes(self, points, selected_indices, parent_indices, pi_prime, 
                    cout_optimal, cost_ff, cost_pi, partitions):
        

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        colors = plt.cm.tab10.colors
        for i, cluster in enumerate(partitions):
            cluster_points = np.array([points[p] for p in cluster])
            plt.scatter(cluster_points[:,0], cluster_points[:,1], 
                    color=colors[i%10], label=f'Cluster {i+1}')
        plt.title(f'Clustering Optimal (Coût: {cout_optimal:.3f})')
        plt.legend()
    
    
        plt.subplot(1, 3, 2)
        G_ff = nx.Graph()
        for i in range(1, len(parent_indices)):
            child = selected_indices[i]
            parent = parent_indices[i]
            num_child = selected_indices.index(child) + 1
            num_parent = selected_indices.index(parent) + 1
            G_ff.add_edge(num_child, num_parent)

        pos_ff = {node: points[selected_indices[node-1]] for node in G_ff.nodes()}
        nx.draw(G_ff, pos_ff, with_labels=True, node_color='lightblue')
        plt.title(f'Farthest-First Tree (Coût: {cost_ff:.3f})')
  
        plt.subplot(1, 3, 3)
        G_pi = nx.Graph()
        for child_num, parent_num in pi_prime.items():
            G_pi.add_edge(child_num, parent_num)
            
    
        pos_pi = {node: points[selected_indices[node-1]] for node in G_pi.nodes()}
        nx.draw(G_pi, pos_pi, with_labels=True, node_color='lightgreen')
        plt.title(f'Arbre Hiérarchique (Coût: {cost_pi:.3f})')
    
        plt.tight_layout()
        plt.show()
    def exporter_vers_csv(self, tree):
       
        try:
         
            from tkinter import filedialog
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")],
                title="Enregistrer les résultats"
            )
        
            if not filepath: 
                return
        
           
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')  
               
                writer.writerow(["Expérience", "Coût FF", "Coût π′", "CoûtOptimal","facteur π′/CoûtOptimal"])
            
               
                for child in tree.get_children():
                    row = tree.item(child)['values']
                    
                    cleaned_row = [
                        str(item).replace(';', ',') if item else '' 
                        for item in row
                    ]
                    writer.writerow(cleaned_row)
                
            messagebox.showinfo("Succès", f"Les résultats ont été exportés vers {filepath}")
        
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'exporter: {str(e)}")
            print(f"Erreur détaillée: {e}")
            
    def afficher_resultats_facteur(self, tableau_final):
        """Affiche les résultats du calcul de facteur"""
        try:
     
            tableau_final['facteur π′/CoûtOptimal'] = pd.to_numeric(tableau_final['facteur  π′/CoûtOptimal'], errors='coerce')
        
            result_window = Toplevel(self.parent)
            result_window.title("Résultats du facteur de coût")
            result_window.geometry("800x600")

    
            main_frame = Frame(result_window)
            main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

         
            summary_frame = Frame(main_frame)
            summary_frame.pack(fill=tk.X, pady=5)
    
            facteur_max = tableau_final['facteur  π′/CoûtOptimal'].max()
            facteur_moyen = tableau_final['facteur π′/CoûtOptimal'].mean()
    
         
            tk.Label(summary_frame, 
                text=f"facteur maximal: {facteur_max:.2f} | facteur moyen: {facteur_moyen:.2f}",
                font=('Arial', 12, 'bold')).pack()

       
            tree_frame = Frame(main_frame)
            tree_frame.pack(fill=BOTH, expand=True)

         
            tree = ttk.Treeview(tree_frame, columns=list(tableau_final.columns), show="headings")
        
        
            for col in tableau_final.columns:
                tree.heading(col, text=col)
                tree.column(col, width=100, anchor='center')

        
            for _, row in tableau_final.iterrows():
                values = [
                    f"{row['Itération']}",
                    f"{float(row['Coût FF']):.2f}" if pd.notna(row['Coût FF']) else "N/A",
                    f"{float(row['Coût π′']):.2f}" if pd.notna(row['Coût π′']) else "N/A",
                    f"{float(row['CoûtOptimal']):.2f}" if pd.notna(row['CoûtOptimal']) else "N/A",
                    f"{float(row['facteur π′/CoûtOptimal']):.2f}" if pd.notna(row['facteur  π′/CoûtOptimal']) else "N/A"
                ]
                tree.insert("", "end", values=values)

     
            vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

       
            tree.grid(row=0, column=0, sticky="nsew")
            vsb.grid(row=0, column=1, sticky="ns")
            hsb.grid(row=1, column=0, sticky="ew")

           
            tree_frame.grid_rowconfigure(0, weight=1)
            tree_frame.grid_columnconfigure(0, weight=1)

        except Exception as e:
            messagebox.showerror("Erreur", f"Problème d'affichage: {str(e)}")
            print(f"Erreur détaillée: {traceback.format_exc()}")

    def quitter_application(self):
       
        if messagebox.askokcancel("Quitter", "Voulez-vous vraiment quitter l'application ?"):
            self.parent.destroy()  


    def executer_experiences(self,nombre_experience,nombre_points,taille_X,taille_Y):
        """Exécute les expériences sans afficher l'interface Choix_random"""
        resultats = Choix_random.calculer_facteur(parent=self)
    
        if resultats is not None:
                    
            self.afficher_resultats_experiences(resultats)
    
    
    def afficher_resultats_experiences(self, resultats):
            
        fenetre_resultats = Toplevel(self)
        fenetre_resultats.title("Résultats des expériences")
        
       
        tree = ttk.Treeview(fenetre_resultats, columns=list(resultats.columns), show="headings")
        
        for col in resultats.columns:
                tree.heading(col, text=col)
        
        for _, row in resultats.iterrows():
                tree.insert("", "end", values=list(row))
        
        tree.pack(expand=True, fill="both")