import os
import re
import shutil
import cv2
import numpy as np
import threading
import torch
import random

def makeVideosIntoFrames(distractedVideosDir,
                     engagedVideosDir, 
                     trainDir,
                     valDir,
                     testDir):
    distractedVideos = os.listdir(distractedVideosDir)
    engagedVideos = os.listdir(engagedVideosDir)
    threads = []
    for i, videoName in enumerate(distractedVideos):
            
        cap = cv2.VideoCapture(f'{distractedVideosDir}/{videoName}')
        
        tNew = threading.Thread(target = makeVideoIntoFrames, args= (cap, i, trainDir, valDir, testDir))
        threads.append(tNew)
    
    for i, videoName in enumerate(engagedVideos):
            
        cap = cv2.VideoCapture(f'{engagedVideosDir}/{videoName}')
        
        tNew = threading.Thread(target = makeVideoIntoFrames, args= (cap, i + 20, trainDir, valDir, testDir))
        threads.append(tNew)
        
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
        


def makeVideoIntoFrames(video, i, trainDir, valDir, testDir):
    
    
    trainImgDir = f'{trainDir}/images'
    trainAnnotDir = f'{trainDir}/labels'
    valImgDir = f'{valDir}/images'
    valAnnotDir = f'{valDir}/labels'
    testImgDir = f'{testDir}/images'
    testAnnotDir = f'{testDir}/labels'
    
    frameNum = 0
    ret, frame = video.read()
    
    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)
    
    while ret:
        if frameNum % 8 == 0:
            
            place = random.random()
            
            imgDir = ''
            annotDir = ''
            
            if place < 0.7:
                imgDir = trainImgDir
                annotDir = trainAnnotDir
            elif place < 0.9:
                imgDir = valImgDir
                annotDir = valAnnotDir
            else:
                imgDir = testImgDir
                annotDir = testAnnotDir
                
                
            frame = cv2.resize(frame, (640, 640), cv2.INTER_AREA)
            frameForModel = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            
            results = model(frameForModel)

            d = False
            
            detections = []
            sizes = []

            for _, r in results.pandas().xyxyn[0].iterrows():
                if r.values[6] == 'person' and r.confidence > 0.4:
                    x1 = r.xmin
                    y1 = r.ymin
                    x2 = r.xmax
                    y2 = r.ymax
                    detections.append([x1, y1, x2, y2])
                    sizes.append((x2-x1)*(y2-y1))
                    d = True
            
            if not d:
                print('failed to find')
            else:
                maxI = np.argmax(sizes)
                detection = detections[maxI]
                
                makeAnnotations(detection, f'{annotDir}/img_{i:03}_{frameNum:07}.txt', frame, i)
                cv2.imwrite(f'{imgDir}/img_{i:03}_{frameNum:07}.png', frame)
            
            
        
        frameNum += 1
        ret, frame = video.read()
    

def makeAnnotations(bbox, fName, bg, i):
    f = open(fName, 'w')
    
    [x1, y1, x2, y2] = bbox
    
    width = (x2 - x1)
    height = (y2 - y1)
    x_center = width/2 + x1
    y_center = height/2 + y1
    
    c = 1 if i < 18 else 0

    f.write(f'{c} {x_center} {y_center} {width} {height}\n')

    f.close()
    
def validate():
    imgs = [cv2.imread('photos/test/images/img_000_0000984.png', -1)]
    annotations = ['photos/test/labels/img_000_0000984.txt']
    for i, img in enumerate(imgs):
        annots = annotations[i]
        img_h, img_w, _ = img.shape
        f = open(annots)
        lines = f.readlines()
        for line in lines:
            vals = line.split(' ')
            x_center = int(float(vals[1]) * img_w)
            y_center = int(float(vals[2]) * img_h)
            width = int(float(vals[3]) * img_w)
            height = int(float(vals[4]) * img_h)
            x_min = x_center - int(width/2)
            x_max = x_center + int(width/2)
            y_min = y_center - int(height/2)
            y_max = y_center + int(height/2)

            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 0, 255))
        cv2.imshow('with rect', img)
        cv2.waitKey(10000)
        
    
def buildTrainingDataAndFiles():
    
    imgBaseFolder = 'photos/'
    if os.path.isdir(imgBaseFolder):
        removeTrainingFileStructure(imgBaseFolder)
    buildTrainingFileStructure()
    
    yamlFile = f'train.yaml'
    if os.path.isfile(yamlFile):
        os.remove(yamlFile)
    createYaml(yamlFile)
        
    makeVideosIntoFrames('distractedVideos',
                     'engagedVideos', 
                     'photos/train',
                     'photos/val',
                     'photos/test')
    
def rmFromDirs(dirs):
    threads = []
    for s in dirs:
        tNew = threading.Thread(target=shutil.rmtree, args=(s, True))
        threads.append(tNew)
        
    for thread in threads:
        thread.start()
    
    for thread in threads:
        thread.join()
    
def removeTrainingFileStructure(dataName):
    rmFromDirs(['photos/train/images',
               'photos/train/labels',
               'photos/val/images',
               'photos/val/labels',
               'photos/test/images',
               'photos/test/labels'])
    
    subDirs = ['photos/train/labels',
            'photos/val/labels',
            'photos/test/labels']
    
    for s in subDirs:
        shutil.rmtree(s, True)
    
    shutil.rmtree(dataName, True)

def buildTrainingFileStructure():
    
    higherpaths = ['photos',
                'photos/train',
                'photos/test',
                'photos/val']
    for path in higherpaths:
        os.mkdir(path)
    
    paths = ['photos/train/images',
               'photos/train/labels',
               'photos/val/images',
               'photos/val/labels',
               'photos/test/images',
               'photos/test/labels']
    for path in paths:
        os.mkdir(path)

def createYaml(yamlFile):
    f = open(yamlFile, 'w')
    
    f.write('path: photos\n')
    f.write(f'train: [train/images]\n')
    f.write(f'val: [val/images]\n')
    f.write(f'test: [test/images]\n')
    f.write('nc: 2\nnames: [ \'Engaged\', \'Distracted\' ]\n')
      
    f.close()

if __name__ == "__main__":
    buildTrainingDataAndFiles()
