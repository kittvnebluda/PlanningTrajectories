from platra.core.map.grid import Grid, cell_array_to_world, grid_to_world, world_to_grid
from platra.core.map.map import CellState
from platra.core.map.prefabs import OCCUPANCY_MATS


class _GridAPI:
    grid = Grid
    to_world = staticmethod(grid_to_world)
    from_world = staticmethod(world_to_grid)
    array_to_world = staticmethod(cell_array_to_world)


grid = _GridAPI()

__all__ = ["grid", "CellState"]
