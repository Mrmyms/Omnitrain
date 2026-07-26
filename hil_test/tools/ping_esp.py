import serial, time
try:
    ser = serial.Serial('/dev/cu.usbmodem101', 115200, timeout=1)
    ser.dtr = False
    ser.rts = False
    time.sleep(1)
    ser.write(b'\n')
    for _ in range(5):
        print(ser.readline().decode('utf-8', errors='ignore').strip())
except Exception as e:
    print(e)
