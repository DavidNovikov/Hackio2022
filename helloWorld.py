import serial


dev = serial.Serial("/dev/tty.usbmodem1101", baudrate=19200)


# dev.write(b'1')
# print(dev.readline())
# dev.write(b'0')
# print(dev.readline())
# dev.write('1'.encode())
# print(dev.readline())
# import serial                                 # add Serial library for Serial communication

# Arduino_Serial = serial.Serial('/dev/tty.usbmodem2101',9600)  #Create Serial port object called arduinoSerialData
# # print(Arduino_Serial.readline())               #read the serial data and print it as line
# print ("Enter 1 to ON LED and 0 to OFF LED")

while 1:                                      #infinite loop
    input_data = input()                  #waits until user enters data
    print("you entered", input_data) 
                       #if the entered data is 0
    dev.write(input_data.encode())             #send 0 to arduino 
    print (f"sending {input_data}")
# import pyfirmata

# import time


# board = pyfirmata.Arduino('/dev/tty.usbmodem2101')


# while True:

#     board.digital[13].write(1)

#     time.sleep(1)

#     board.digital[13].write(0)

#     time.sleep(1)