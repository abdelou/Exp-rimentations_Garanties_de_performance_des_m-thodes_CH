from __future__ import annotations
from typing import List, Optional


class Point:
    def __init__(
        self,
        x: float,
        y: float,
        nombre: int,
        sonPere: Optional["Arret"] = None,
        lesEnfant: Optional[List["Point"]] = None,
        cluster: Optional["Cluster"] = None,
        niveau: int = 0
    ):
        self.x = float(x)
        self.y = float(y)
        self.nombre = int(nombre)
        self.sonPere = sonPere
        self.lesEnfant = lesEnfant if lesEnfant is not None else []
        self.cluster = cluster
        self.niveau = int(niveau)

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y}, nombre={self.nombre}, niveau={self.niveau})"
