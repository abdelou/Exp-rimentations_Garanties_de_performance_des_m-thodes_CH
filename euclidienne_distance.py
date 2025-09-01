#ici dans ce classe on calculer la fonction de distance euclidean 

from distance import Distance
from point import Point
class Euclidienne_Distance(Distance):
    
    #calculer la distance euclidienne entre deux points dans l'espace metric
    def calcule_distance(self,p1,p2):
        if isinstance(p1, Point) and isinstance(p2, Point):
            x1, y1 = p1.x, p1.y
            x2, y2 = p2.x, p2.y

        else:
            if isinstance(p1, str):
                x1, y1 = map(float, p1.split(','))
            else:
                x1, y1 = p1

            if isinstance(p2, str):
                x2, y2 = map(float, p2.split(','))
            else:
                x2, y2 = p2
    
        return (((x1 - x2) ** 2) + ((y1 - y2) ** 2)) ** 0.5