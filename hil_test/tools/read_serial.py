import serial
import time

try:
    ser = serial.Serial('/dev/cu.usbmodemD40592796EE41', 115200, timeout=1)
    ser.dtr = False
    ser.rts = False
    print("Connected. Listening...")
    while True:
        line = ser.readline()
        if line:
            print(line.decode('utf-8', errors='ignore').strip())
except Exception as e:
    print(e)
