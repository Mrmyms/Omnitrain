__version__ = '1.2.0'

from .robot_registry import RobotRegistry, OmniBaseRobot, SensorSpec, register_robot
import omnitrain.robots # Auto-register built-in robots

from omnitrain.sdk import ProjectManager, LiquidTrainer, EdgeDeployer, AgentRunner
from omnitrain.fusion_core import LiquidFusionCore
from omnitrain.esp32_exporter import ESP32Exporter
from omnitrain.jetson_exporter import JetsonExporter
from omnitrain.serial_logger import ESP32SerialLogger

