
from __future__ import annotations
class Cluster:
    def __init__(self, centre: Point, cost: float):
        self.centre = centre
        self.cost = float(cost)

    def __repr__(self):
        return f"Cluster(centre={self.centre}, cost={self.cost})"
