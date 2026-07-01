__version__ = '1.0.0'

from .robot_registry import RobotRegistry, OmniBaseRobot, SensorSpec, register_robot
import omnitrain.robots # Auto-register built-in robots

from omnitrain.sdk import ProjectManager, LiquidTrainer, EdgeDeployer, AgentRunner
