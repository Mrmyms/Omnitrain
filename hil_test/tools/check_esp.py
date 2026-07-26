import serial, time
ser = serial.Serial('/dev/cu.usbmodemD40592796EE41', 115200, timeout=2)
ser.setDTR(True)
ser.setRTS(False)
time.sleep(1)
ser.write(b"\n\n999.0\n")
time.sleep(1)
print("REPLY:", ser.read_all().decode('utf-8', errors='ignore'))
