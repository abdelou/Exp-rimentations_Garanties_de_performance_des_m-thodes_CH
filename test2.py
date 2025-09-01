import matplotlib.pyplot as plt


def dessiner_parallelepipede():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Points du carré principal (non joint sur le côté droit)
    carre = [
        [0, 0],  # A
        [2, 0],  # B
        [2, 2],  # C
        [0, 2]   # D
    ]
    
    # Dessiner les côtés (sauf le côté droit qui sera ouvert)
    ax.plot([carre[0][0], carre[1][0]], [carre[0][1], carre[1][1]], 'b-')  # AB
    ax.plot([carre[1][0], carre[2][0]], [carre[1][1], carre[2][1]], 'b--') # BC (en pointillés)
    ax.plot([carre[2][0], carre[3][0]], [carre[2][1], carre[3][1]], 'b-')  # CD
    ax.plot([carre[3][0], carre[0][0]], [carre[3][1], carre[0][1]], 'b-')  # DA
    
    # Ajouter des annotations pour les points
    points_labels = ['A', 'B', 'C', 'D']
    for i, point in enumerate(carre):
        ax.plot(point[0], point[1], 'ro')  # Points rouges
        ax.text(point[0]+0.1, point[1]+0.1, points_labels[i], fontsize=12)
    
    # Paramètres d'affichage
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title('Parallélépipède en 2D (côté droit non joint)')
    ax.grid(True)
    
    plt.show()

dessiner_parallelepipede()