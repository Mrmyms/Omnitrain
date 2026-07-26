import serial, time
port = "/dev/cu.usbmodem101"
# We must open without asserting DTR/RTS initially
ser = serial.Serial()
ser.port = port
ser.dtr = False
ser.rts = False
ser.open()

print("Resetting into normal mode...")
# EN = Low (RTS = True), BOOT = High (DTR = False)
ser.setRTS(True)
ser.setDTR(False)
time.sleep(0.1)

# EN = High (RTS = False), BOOT = High (DTR = False)
ser.setRTS(False)
ser.setDTR(False)
time.sleep(1.0)
