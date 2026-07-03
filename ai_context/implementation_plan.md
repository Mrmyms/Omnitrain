# Peer Review Overhaul Plan

Tu análisis es **impecable y dolorosamente certero**. Un comité de TinyML o EMNLP destrozaría el artículo actual por confundir la ventaja matemática (CfC) con la ventaja de ingeniería (Zero-Copy) y por usar un tono de "white paper" comercial. 

Para elevar este documento a un verdadero estándar académico (IEEE/ACM), propongo el siguiente plan de ataque en dos frentes:

## 1. Rediseño Experimental y Ablation (Código Python)
Debemos aislar las variables y agregar rigor estadístico a nuestras afirmaciones.
*   **Múltiples Semillas (Varianza):** Actualizaré `train_and_compare.py` para entrenar y evaluar los modelos sobre **5 semillas aleatorias (random seeds) distintas**. El script calculará y guardará la Media $\pm$ Desviación Estándar del MSE, eliminando la sospecha de *cherry-picking*.
*   **Ablation (LSTM Zero-Copy Emulado):** Añadiré un baseline teórico de `LSTM (Zero-Copy)` a nuestra tabla de hardware. Esto admitirá abiertamente que si pasáramos un LSTM por nuestro OmniEngine, el SRAM también bajaría a $\approx 1.5$ KB. Así demostramos honestidad intelectual: aceptamos que la ganancia de memoria viene de la ingeniería, pero defendemos que la **resiliencia al jitter (pérdida de paquetes)** viene exclusivamente de la matemática CfC.
*   **Actualización de Gráfica:** Generaré el nuevo `temporal_resilience_chart.png` incluyendo las barras de error (desviación estándar) y lo incrustaré oficialmente en el documento LaTeX con `\includegraphics`.

## 2. Reestructuración y "Des-marketing" del LaTeX
Voy a purgar el lenguaje promocional y a reestructurar la narrativa científica en `paper_draft.tex`:
*   **Moderar el Tono:** Eliminaré frases como *"extreme robustness"*, *"catastrophic degradation"*, y *"drastically reduced"*. El número "0.00" de MSE será reemplazado por la medición científica real de la paridad en punto flotante ($MSE < 10^{-6}$), explicando que se debe al estándar IEEE 754.
*   **Detalles Físicos y de Energía:** Aclararé que la protoboard funcionó a 80MHz QIO, pero eliminaré la afirmación de que eso prueba "robustez extrema". También especificaré que la medición de consumo (INA219) excluyó explícitamente el radio WiFi/Bluetooth del ESP32-S3 para aislar el consumo del núcleo Xtensa.
*   **[NEW] Sección de Limitaciones:** Agregaré una sección obligatoria de *Limitations* antes del *Future Work*. Aquí admitiremos tres cosas:
    1. Que la comparación inicial con TFLite confunde arquitectura vs framework (y referenciamos nuestro ablation teórico).
    2. Que los accesos no alineados a la memoria Flash SPI (misaligned memory access) pueden generar cuellos de botella no medidos en arquitecturas mayores.
    3. Que las pruebas actuales son *Processor-in-the-Loop* (simuladas) y no *Hardware-in-the-Loop* físico.

## 3. [NUEVO] Pruebas Físicas "Hardware-in-the-Loop" (ESP32)
Ya que vas a conectar un ESP32 clásico (Xtensa LX6, 520KB SRAM), podemos hacer algo increíble: **elevar el paper de Processor-in-the-Loop (PiL) a Hardware-in-the-Loop (HIL) real**.
El plan es:
1. Instalar PlatformIO Core (`pio`) en el entorno virtual.
2. Crear un proyecto firmware `esp32dev` e inyectarle el código fuente de tu `OmniEngine.cpp`.
3. Compilar y flashear (quemar) el firmware directamente a tu ESP32 por USB.
4. Escribir un script Python (`run_hil_test.py`) que se comunique por puerto Serial (PySerial) con el ESP32, enviándole el dataset de Péndulo Invertido con jitter, y leyendo las predicciones reales calculadas por el silicio para validar que el MSE se mantiene estable en hardware físico.

## User Review Required
> [!IMPORTANT]
> **Pruebas Físicas (HIL):**
> Si apruebas este plan, necesitaré que conectes tu módulo ESP32 por USB a tu Mac. 
> Cuando presiones **Proceed**, instalaré PlatformIO, compilaré el código C++ (`OmniEngine.cpp`), lo flashearé a la placa y ejecutaré un script de validación que enviará los datos del péndulo por Serial (USB) y leerá las predicciones del procesador real en tiempo real. 
> 
> ¿Estás listo? ¡Conecta el ESP32 y dale a **Proceed**!
>
> **Preguntas adicionales:**
> 1. Para aislar las variables, ¿estás de acuerdo en que agreguemos la fila teórica de "LSTM (Zero-Copy)" a la Tabla I, asumiendo un SRAM similar al CfC, para demostrar honestidad académica?
> 2. ¿Quieres que yo redacte directamente la sección de **Limitaciones** en el LaTeX siguiendo los puntos de arriba, o tienes algún otro cuello de botella técnico del ESP32-S3 que quieras agregar a esa sección (ej. cuellos de botella en el bus DMA)?
> 3. En cuanto a las gráficas, ¿te parece bien si actualizo el script para incluir barras de error ($\pm \sigma$) a lo largo de las 5 semillas aleatorias?

Espero tu retroalimentación sobre estas preguntas para empezar a reescribir la historia y aislar científicamente nuestra contribución.
