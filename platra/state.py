from pygame import Vector2


class Twist:
    def __init__(self, vx: float = 0, vy: float = 0, w: float = 0) -> None:
        self.v = Vector2(vx, vy)
        self.w = w


class Pose:
    def __init__(self, position: Vector2, orientation: float = 0) -> None:
        self.pos = position
        self.theta = orientation


class GameState:
    def __init__(self, pose: Pose, velocity: Twist) -> None:
        self.pose = pose
        self.vel = velocity
        self.dt = 0.0
        self.running = True
