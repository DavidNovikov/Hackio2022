from os import stat
import cv2
import torch
import time
import serial


dev = serial.Serial("/dev/tty.usbmodem1101", baudrate=19200)



# Model
model_path = 'best.pt'
model_conf = 0.65
model = torch.hub.load(
    'yolov5', 'custom', path=model_path, source='local')
model.conf = model_conf

def goToEnaged():
    dev.write('3'.encode()) # alarms off
    dev.write('4'.encode()) # blue on

def firstDistracted():
    dev.write('1'.encode()) # red on
    dev.write('5'.encode()) # blue off

def secondDistracted():
    dev.write('2'.encode()) # buzzer on
    dev.write('5'.encode()) # blue off

def thirdDistracted():
    dev.write('0'.encode()) # vibrate on
    dev.write('5'.encode()) # blue off
    

def video():
    
    states = []
    
    video = cv2.VideoCapture(1)
    ret, frame = video.read()
    timeDistracted = 0
    # 0 - engaged
    # 1 - distracted
    while ret:
        frame = cv2.resize(frame, (640, 640), cv2.INTER_AREA)
        frameForModel = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frameForModel)

        d = False

        currentTime = time.time()
        for _, r in results.pandas().xyxyn[0].iterrows():
            if r.confidence > 0.1:
                states.append([r.values[6],currentTime])
        
        
        states = [s for s in states if s[1] > currentTime - 1]
        
        predict = sum([int(s[0]) for s in states])/ (len(states) + 1) #avoid divide by zero
        
        if predict > 0.5:
            timeDistracted = timeDistracted + (states[-1][1] - states[-2][1])
        else:
            timeDistracted = 0
            # goToEnaged()
            print('good driving')
        
        if timeDistracted > 5:
            # thirdDistracted()
            print('3rd dist')
        if timeDistracted > 3:
            # secondDistracted()
            print('2rd dist')
        if timeDistracted > 1:
            # firstDistracted()
            print('1rd dist')
        
            

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, frame = video.read()

def test():
    goToEnaged()
    time.sleep(5)
    
    firstDistracted()
    time.sleep(1)
    
    secondDistracted()
    time.sleep(1)
    
    thirdDistracted()
    time.sleep(3)
    
    goToEnaged()
    time.sleep(1)

# test()
video()