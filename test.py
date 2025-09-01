import tkinter as tk
from tkinter import ttk
from tkinter import Label, Entry, Frame

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Interface de création de formes géométriques")
        self.geometry("1500x850")
        
        
        self.cadre_principal = Frame(self)
        self.cadre_principal.pack(expand=True, fill=tk.BOTH)
        
        
        self.cadre_controles = Frame(self.cadre_principal)
        self.cadre_controles.pack(fill=tk.X, padx=5, pady=5)
        
        # ComboBox pour le choix de la forme géométrique
        Label(self.cadre_controles, text="Forme géométrique:").grid(row=0, column=0, padx=5)
        self.choix_forme = ttk.Combobox(self.cadre_controles, 
                                      values=["Carré plein", "Diagonale droite", "Diagonale gauche", 
                                              "Cercle plein", "Droite horizontale", "Droite Verticale"])
        self.choix_forme.grid(row=0, column=1, padx=5)
        self.choix_forme.set("Choisir une forme") 
        
        #   le nombre de points
        Label(self.cadre_controles, text="Nombre de points:").grid(row=0, column=2, padx=5)
        self.entree_points = Entry(self.cadre_controles, width=5)
        self.entree_points.grid(row=0, column=3, padx=5)
        self.entree_points.insert(0, "100")  
        
        #   la taille  de l'axe X
        Label(self.cadre_controles, text="Taille axe X:").grid(row=0, column=4, padx=5)
        self.entree_axe_x = Entry(self.cadre_controles, width=5)
        self.entree_axe_x.grid(row=0, column=5, padx=5)
        self.entree_axe_x.insert(0, "10")  
        
        #   la taille l'axe Y
        Label(self.cadre_controles, text="Taille axe Y:").grid(row=0, column=6, padx=5)
        self.entree_axe_y = Entry(self.cadre_controles, width=5)
        self.entree_axe_y.grid(row=0, column=7, padx=5)
        self.entree_axe_y.insert(0, "10")  
        
        # Bouton pour créer la forme
        ttk.Button(self.cadre_controles,
                  text="Créer la forme géométrique",
                  command=self.creer_forme).grid(row=0, column=8, padx=5)
        
        # Bouton pour calculer le facteur
        ttk.Button(self.cadre_controles,
                  text="Calculer Facteur",
                  command=self.calculer_facteur_optimal).grid(row=0, column=9, padx=5)
        
        # Bouton pour initialiser l'espace
        ttk.Button(self.cadre_controles,
                  text="Initialiser l'espace",
                  command=self.initialiser_espace).grid(row=0, column=10, padx=5)
        
        # Bouton pour Quitter
        ttk.Button(self.cadre_controles,
                  text="Quitter",
                  command=self.retour_choix).grid(row=0, column=11, padx=5)
        
        # ici l'espace métrique pour afficher des  formes
        self.frame_espace = Frame(self.cadre_principal, bg='white')
        self.frame_espace.pack(expand=True, fill=tk.BOTH, pady=20)
    
    def creer_forme(self):
        forme_selectionnee = self.choix_forme.get()
        nb_points = self.entree_points.get()
        taille_x = self.entree_axe_x.get()
        taille_y = self.entree_axe_y.get()
        print(f"Création d'un {forme_selectionnee} avec {nb_points} points, taille X={taille_x}, Y={taille_y}")
       
    def calculer_facteur_optimal(self):
        print("Calcul du facteur optimal en cours...")
    
    def initialiser_espace(self):
        print("Initialisation de l'espace...")
       
    def retour_choix(self):
        self.quit()

if __name__ == "__main__":
    app = Application()
    app.mainloop()