# Auto-registers all built-in robots via side-effect imports.
# Each module uses @register_robot to add itself to RobotRegistry on import.
from omnitrain.robots import sim_delivery_bot  # noqa: F401
