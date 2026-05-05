# Guía de Conectividad: Cómo conectar sensores a OmniTrain

Esta guía explica los 4 métodos disponibles para alimentar de datos (inputs) al cerebro de OmniTrain. Independientemente del método, todos los datos terminan en el `TokenBus` (Shared Memory) para ser procesados por el solver Liquid Neural Network.

---

## Requisito Previo: Configuración del YAML (`config.yaml`)
Antes de conectar cualquier sensor, debes definir su "espacio" en `config.yaml`. El sistema lee este archivo y reserva memoria automáticamente (Shared Memory) para cada sensor. El parámetro `id` debe coincidir exactamente con el que uses en el código del plugin.

```yaml
inputs:
  - id: "my_sensor_name"
    type: "sensor"       # Opciones: "sensor" (vectorial), "vision" (imágenes), "boolean" (banderas)
    hz: 10               # Frecuencia de actualización en Hercios. Importante para el cálculo del tiempo continuo (dt).
    dim: 512             # Dimensión del token final. Se recomienda 512 para estandarizar el 'Conectoma'.
    range: [0.0, 5.0]    # Rango de valores físicos (mín, máx). Útil para que la IA sepa los límites.
    noise: true          # Si es true, la red neuronal aplicará "Curriculum Dropout" asumiendo que el sensor puede fallar.
```

---

## Creando un Plugin Personalizado (Custom Sensors)
Si tienes un sensor no estándar (e.g. un arreglo de micrófonos o sensores táctiles I2C), puedes crear tu propio puente heredando de la clase base `ModalityPlugin`. El único requisito es que transformes tus datos crudos a un vector `Numpy` del tamaño definido en `dim`.

```python
import numpy as np
from omnitrain.plugins import ModalityPlugin

class MiSensorPersonalizado(ModalityPlugin):
    def read_raw_data(self):
        # 1. Lógica de hardware: Leer del puerto Serie, I2C, SPI, etc.
        # EJEMPLO: return spi.read()
        return [2.5, 3.1, 0.4] 

    def encode(self, raw_data):
        # 2. Pre-procesamiento y Normalización
        # El sistema espera un vector (token) del tamaño exacto definido en el yaml (ej. 512)
        token = np.zeros(512, dtype='float32')
        
        # Insertar los datos y normalizar al rango [0, 1] si es posible
        token[0] = raw_data[0] / 5.0
        token[1] = raw_data[1] / 5.0
        token[2] = raw_data[2] / 5.0
        
        return token

# Para ejecutarlo:
# plugin = MiSensorPersonalizado(bus, "my_sensor_name", hz=10)
# plugin.run()
```

---

## 1. Método Offline / Datos Reales (`plugins_real.py`)
Ideal para pruebas rápidas sin hardware o para re-entrenar con grabaciones.

### Uso de CSV
Si tienes datos en una tabla:
1. Asegúrate de que las columnas del CSV coincidan con tu sensor.
2. Usa `CSVModalityPlugin`.

```python
from omnitrain.plugins_real import CSVModalityPlugin
plugin = CSVModalityPlugin(bus, "my_sensor_name", hz=10, csv_path="data.csv")
plugin.run()
```

### Uso de Carpetas de Imágenes
Para simular una cámara con fotos locales:
```python
from omnitrain.plugins_real import ImageFolderPlugin
plugin = ImageFolderPlugin(bus, "my_vision", hz=5, img_dir="./frames")
plugin.run()
```

---

## 2. Método Robótica (ROS 2)
El estándar para robots físicos (Humble/Iron/Jazzy). OmniTrain usa un patrón "Singleton" interno para el Nodo ROS 2, lo que significa que puedes crear decenas de plugins sin colisionar con el manejador de memoria `rclpy`.

1. **Asegúrate de que ROS 2 esté en tu path** (`source /opt/ros/...`).
2. **Si usas mensajes personalizados**, asegúrate de construir (build) y hacer source de tu workspace local de ROS 2.
3. **Lanza los Plugins** instanciando los ya preconstruidos, o hereda de `ROS2BasePlugin` si necesitas un mensaje distinto a Image/LaserScan.

```python
from omnitrain.plugins_ros2 import ROS2CameraPlugin, ROS2LidarPlugin

# Conectar Cámara (transforma automáticamente sensor_msgs/Image a un Token 512-dim)
cam = ROS2CameraPlugin(bus, "front_cam", hz=30, topic_name="/camera/image_raw")
cam.run() # IMPORTANTE: run() bloquea el hilo, si usas varios plugins, lánzalos en hilos o procesos separados.

# Conectar Lidar (limpia NaNs/Infs y sub-muestrea a 512-dim automáticamente)
lidar = ROS2LidarPlugin(bus, "laser_scan", hz=10, topic_name="/scan")
lidar.run()
```

---

## 3. Método Simulación (NVIDIA Isaac Sim)
Para entrenar "Gemelos Digitales" y usar Reinforcement Learning en Omniverse. Este bridge está altamente optimizado para bajo consumo de RAM (ideal para estaciones de trabajo con 16GB VRAM, como una RTX 5070).

1. Inicia el entorno Python nativo de Isaac Sim (`python.sh`).
2. El `IsaacOmniBridge` carga el robot, lanza el simulador de físicas de la GPU y puentea los datos de los sensores virtuales (e.g. `LidarRtx`) directamente al `TokenBus`.
3. También maneja el **bucle de retroalimentación cerrado**: extrae las acciones que produce la IA y las envía a los `ArticulationAction` de los motores en la simulación.

```python
from omnitrain.isaac_bridge import IsaacOmniBridge

# token_dim debe ser idéntico al 'dim' de config.yaml
bridge = IsaacOmniBridge(session_id="isaac_train", robot_name="mi_robot")
bridge.setup_scene(robot_usd_path="/rutas/a/mi_robot.usd") 
# La simulación comenzará a transmitir en tiempo real.
```

---

## 4. Método Hardware Distribuido
Para sistemas con doble procesador (e.g., Qualcomm para IA + STM32 para Control).

1. **AI Brain**: Procesa la red neuronal y envía "intenciones".
2. **Action Brain**: Recibe las intenciones y aplica el `OmniShieldGuard`.

```python
from omnitrain.edgecp_bridge import DualBrainRPC, EdgeCPAIBrain, EdgeCPActionBrain

rpc = DualBrainRPC()
ai = EdgeCPAIBrain(rpc)
action = EdgeCPActionBrain(rpc, my_shield)

action.start() # Bucle a 1000Hz
ai.start()     # Bucle a 30Hz
```

---

## Consejos de Oro
- **Normalización**: Asegúrate de que en `encode()`, tus datos siempre terminen entre `0.0` y `1.0` o `-1.0` y `1.0`. Las redes Liquid son sensibles a la escala.
- **Asincronía**: Los plugins corren en sus propios hilos/procesos. No bloquees el hilo principal de ROS.
- **Diagnóstico**: Usa el comando `/bus` en el CLI para verificar en tiempo real si los datos están llegando al `TokenBus`.
