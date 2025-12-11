import numpy as np
from pygame import Vector2


def np_array_to_vectors(array: np.ndarray) -> list[Vector2]:
    return [Vector2(x, y) for x, y in array]
