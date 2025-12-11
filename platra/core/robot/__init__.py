from typing import Optional

from .base import RobotConfig, RobotModel, RobotState
from .configs import AckermannConfigForStaticFeedback


def create_robot_model(
    conf: RobotConfig, initial_state: Optional[RobotState] = None
) -> RobotModel:
    return RobotModel(conf, initial_state)
