from .grid import Grid, cell_array_to_world, grid_to_world, world_to_grid
from .map import CellState
from .prefabs import OCCUPANCY_MATS


class _GridAPI:
    grid = Grid
    to_world = staticmethod(grid_to_world)
    from_world = staticmethod(world_to_grid)
    array_to_world = staticmethod(cell_array_to_world)


grid = _GridAPI()

__all__ = ["grid", "CellState"]
