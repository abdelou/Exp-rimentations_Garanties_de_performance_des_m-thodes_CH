import numpy as np
from math import inf
from euclidienne_distance import Euclidienne_Distance
class Farthest_First_Traversal:
    def __init__(self, beta=2, R=1.0):
        self.beta = beta 
        self.R = R       
        self.distance_calculator = Euclidienne_Distance() 

    def lev(self, p, r):
        """Calcule le niveau hiérarchique d'un point en fonction de son rayon r."""
        j = 1
        while (self.R / (self.beta ** (j - 1)) >= r) and (r > self.R / (self.beta ** j)):
            j += 1
        return j

    def PI(self, P, px, min_S):
        """Trouve le point le plus éloigné et son parent."""
        p_f, p_p = None, None
        greatest_low_distance = -inf

        for j in range(len(P)):
            p_j = P[j]
            r = self.distance_calculator.calcule_distance(px, p_j)

            if r < min_S[j][0]:
                min_S[j][0] = r
                min_S[j][1] = px

            if min_S[j][0] > greatest_low_distance:
                greatest_low_distance = min_S[j][0]
                p_f = p_j
                p_p = min_S[j][1]

        return p_f, p_p

    def Pl_prime(self, S, i):
        """Trouve le parent d'un point dans le niveau de granularité le plus bas possible."""
        parent = None
        candidate_r = inf

        for j in range(i - 1, -1, -1):
            r = self.distance_calculator.calcule_distance(S[j], S[i])
            lev_j = self.lev(S[j], r)
            lev_i = self.lev(S[i], r)

            if r <= candidate_r and lev_j < lev_i:
                candidate_r = r
                parent = S[j]

        return parent

    def traverse(self, P):
        """Algorithme principal de Farthest-First Traversal."""
        S = [] 
        n = len(P)

        px = P[np.random.choice(range(n))]
        S.append(px)
        P.remove(px)

        min_S = [[inf, None] for _ in range(n)]

    
        for i in range(1, n):
            px = S[-1]
            p_f, p_p = self.PI(P, px, min_S)

            if p_f is None:
                break 

            
            S.append(p_f)
            P.remove(p_f)

        return S