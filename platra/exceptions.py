class Platra(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


class BadCubicParabolaParameter(Platra):
    def __init__(
        self,
        cubic_parabola_coef: float | int,
        min_corner_radius: float | int,
        minimal_radius: float | int,
    ) -> None:
        super().__init__(f"""Too small radius: {min_corner_radius} for the cubic 
parabola with coeffitient {cubic_parabola_coef}, increase parabola 
coeffitient to build a corner of this radius.
Minimal radius for the coeffitient is {minimal_radius: 0.3f}.
Minimal coeffitient for the given radius is {18 / (25 * min_corner_radius**2 * 5 ** (1 / 2)): 0.3f}""")
