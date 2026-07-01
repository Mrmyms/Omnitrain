__version__ = '2.2.0'

from .robot_registry import RobotRegistry, OmniBaseRobot, SensorSpec, register_robot
import omnitrain.robots # Auto-register built-in robots
