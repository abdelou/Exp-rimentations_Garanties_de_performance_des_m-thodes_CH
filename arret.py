from point import Point

class Arret:
    def __init__(self, parent: Point, enfant: Point):
        self.parent = parent
        self.enfant = enfant

    def __repr__(self):
        return f"Arret(parent={self.parent}, enfant={self.enfant})"
