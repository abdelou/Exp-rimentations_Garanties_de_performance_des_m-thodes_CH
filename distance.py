#ici on creer une classe interface est souvant on dit 
#une classes abstraites donc ici en python 
#on utlise le module ABC c'est a dire abstract Base Classes 
#qui permet de defini les classes et les methodes abstraites 


from abc import ABC, abstractmethod

class Distance(ABC): #ici distance herite de ABC ce qui signifie que c'est une classe abstraite
    @abstractmethod #celui la dit que tous les classe qui hérite de Dsitance 
    #doit implemetnter obliger cette methode

    def calcule_distance(self,p1,p2):# ce methode 
        pass
