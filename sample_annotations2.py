import math
import shutil
import cv2
import random
import numpy as np
import os
from PIL import Image
import threading
import bigText

dronesTestDir = 'photos/drones/testAug'
dronesTrainDir = 'photos/drones/trainAug'
bgTestingDir = 'photos/background/testingBgs'
bgTrainingDir = 'photos/background/trainingBgs'
testingAnnotationsDir = 'photos/generatedData/testing/labels'
testingImagesDir = 'photos/generatedData/testing/images'
trainingAnnotationsDir = 'photos/generatedData/training/labels'
trainingImagesDir = 'photos/generatedData/training/images'
validationDir = 'photos/generatedData/validation/images'

smallSampleBgsDir = 'photos/background/smallSample'
smallSampleDronesDir = 'photos/drones/smallSampleAug'
smallSampleAnnotationsDir = 'photos/generatedData/smallSample/labels'
smallSampleImageDir = 'photos/generatedData/smallSample/images'

index = 0
tot_width = 0
tot_height = 0
numthreads = 35

minWidth = 9
maxWidth = 55
tot_width_test = 0
tot_height_test = 0
numthreads = 40

def scaleDrone(drone, bg_w, minWidth=minWidth, maxWidth=maxWidth):
    # drone_w, drone_h= drone.size
    
    # maxWidth = min(bg_w, maxWidth)
    # minWidth = int(math.sqrt(100*drone_w/drone_h))
    # valInRange = np.random.randint(minWidth, maxWidth)
    
    # newWidth = int((valInRange**3/maxWidth**2.11)*(maxWidth-minWidth)/maxWidth+minWidth)
    
    # newHeight = int((drone_h * newWidth) / float(drone_w))

    # newDrone = drone.resize((newWidth, newHeight), 2)# 2 is pil.image.bilinear
    # return newDrone
    
    maxWidth = min(bg_w, maxWidth)
    drone_w, drone_h= drone.size
    
    valInRange = np.random.randint(minWidth, maxWidth)
    newWidth = int((valInRange**3/maxWidth**2)*(maxWidth-minWidth)/maxWidth+minWidth)
    
    newHeight = int((drone_h * newWidth) / float(drone_w))

    newDrone = drone.resize((newWidth, newHeight), 2)
    return newDrone


def intersectsWithExisting(drone_h, drone_w, x, y, bboxes):
    intersecting = False
    xmin = x
    ymin = y
    xmax = x + drone_w
    ymax = y + drone_h
    for box in bboxes:
        intersecting |= intersects(
            xmin, ymin, xmax, ymax, box['xmin'], box['ymin'], box['xmax'], box['ymax'])

    return intersecting


def intersects(x1min, y1min, x1max, y1max, x2min, y2min, x2max, y2max):
    return x1max >= x2min and x2max >= x1min and y1max >= y2min and y2max >= y1min


def giveNewBoundingBox(drone_h, drone_w, bboxes, bg_h, bg_w):
    x = np.random.randint(0, bg_w - drone_w)
    y = np.random.randint(0, bg_h - drone_h)
    while intersectsWithExisting(drone_h, drone_w, x, y, bboxes):
        x = np.random.randint(0, bg_w - drone_w)
        y = np.random.randint(0, bg_h - drone_h)
    return x, y


def pasteDronesOnBackGround(drones, bg):
    bg_w, bg_h = bg.size
    bboxes = []
    for drone in drones:
        scaledDrone = scaleDrone(drone, bg_w/3)
        drone_w, drone_h = scaledDrone.size

        x, y = giveNewBoundingBox(drone_h, drone_w, bboxes, bg_h, bg_w)
        newbox = {'xmin': x, 'xmax': x+drone_w, 'ymin': y, 'ymax': y+drone_h}
        bboxes.append(newbox)
        
        bg.paste(scaledDrone, (x,y), scaledDrone)

    return bboxes

def makeAnnotations(bboxes, fName, bg):
    bg_w, bg_h = bg.size
    f = open(fName, 'w')
    total_width = 0
    total_height = 0
    for box in bboxes:
        # get pixel values
        width = (box['xmax'] - box['xmin'])
        height = (box['ymax'] - box['ymin'])
        x_center = width/2 + box['xmin']
        y_center = height/2 + box['ymin']
        
        total_width += width
        total_height += height

        # scale
        width = width / bg_w
        height = height / bg_h
        x_center = x_center / bg_w
        y_center = y_center / bg_h

        f.write(f'0 {x_center} {y_center} {width} {height}\n')

    f.close()
    
    return total_width, total_height


def makeTrainingData(dDir, bgDir, imgDir, annotDir):
    drones = os.listdir(dDir)
    numDrones = len(drones)
    random.shuffle(drones)

    bgs = os.listdir(bgDir)
    numBgs = len(bgs)

    prob = 2 * numDrones/numBgs

    print('numDrones', numDrones)
    print('numBgs', numBgs)
    print('prob', prob)
    
    global tot_height
    global tot_width
    global index
    
    tot_width = 0
    tot_height = 0
    index = 0
    
    aggLock = threading.Lock()
    indexLock = threading.Lock()
    
    threads = []
    for i in range(numthreads):
        low = int(i * numBgs/numthreads)
        high = int((i+1) * numBgs/numthreads)
        newT = threading.Thread(target=makeTrainingDataForSubSet, args=(bgs[low:high], bgDir, prob, drones, numDrones, 
                                                                        dDir, annotDir, imgDir, aggLock, indexLock, low))
        threads.append(newT)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    
    
    
    print('avg drone width', tot_width / index)
    print('avg drone height', tot_height / index)
    print('drones made', index)

def makeTrainingDataForSubSet(bgs, bgDir, prob, drones, numDrones, dDir, annotDir, imgDir, aggLock, indexLock, low):
    
    global tot_height
    global tot_width
    global index
    
    for b, bgName in enumerate(bgs):
        bg = Image.open(os.path.join(bgDir, bgName))
        dronesForImg = []
        
        while random.random() < prob and len(dronesForImg) < 4:
            localIndex = np.NaN
            
            indexLock.acquire() 
            localIndex = index
            index += 1
            indexLock.release()
            
            droneName = drones[localIndex % numDrones]
            
            drone = Image.open(os.path.join(dDir, droneName))
            dronesForImg.append(drone)
            
        bboxes = pasteDronesOnBackGround(dronesForImg, bg)

        widths, heights = makeAnnotations(
            bboxes, f'{annotDir}/img{b+low:05}.txt', bg)
        
        aggLock.acquire()
        tot_width += widths
        tot_height += heights
        aggLock.release()
        
        
        bg.save(f'{imgDir}/img{b+low:05}.png')


def splitIntoTrainAndValidation(trainingDir, validationDir):
    data = os.listdir(trainingDir)
    dataArr = np.asarray(data)
    print('Original Number of images', len(dataArr))
    mask = np.random.rand(len(dataArr)) < 0.77
    training_data = dataArr[mask]
    validation_data = dataArr[~mask]

    print(f"No. of training examples: {training_data.shape}")
    print(f"No. of validation examples: {validation_data.shape}")

    for img in validation_data:
        shutil.move(
            f'{trainingDir}/{img}', f'{validationDir}/{img}')
        num = int(img[3:8])
        shutil.move(
            f'{trainingDir}/../labels/img{num:05}.txt',
            f'{validationDir}/../labels/img{num:05}.txt')


def makeGreyScaleCopy(srcImgDir, destImgDir, srcLabelDir, destLabelDir):
    imgs = os.listdir(srcImgDir)
    labels = os.listdir(srcLabelDir)
    numBgs = len(imgs)
    
    
    threads = []
    for i in range(numthreads*2):
        low = int(i * numBgs/(numthreads*2))
        high = int((i+1) * numBgs/(numthreads*2))
        imgSubset = imgs[low:high]
        newT = threading.Thread(target=makeGreyScaleImgCopySubSet, args=(imgSubset,srcImgDir, destImgDir))
        threads.append(newT)
        labelSubset = labels[low:high]
        newT = threading.Thread(target=makeGreyScaleLabelCopySubSet, args=(labelSubset,srcLabelDir, destLabelDir))
        threads.append(newT)
    
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
        
def makeGreyScaleImgCopySubSet(imgs, srcImgDir, destImgDir):
    for imgName in imgs:
        img = cv2.imread(os.path.join(srcImgDir, imgName), -1)
        grey = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        cv2.imwrite(os.path.join(destImgDir, imgName), grey)


def makeGreyScaleLabelCopySubSet(labels,srcLabelDir, destLabelDir):
    for label in labels:
        shutil.copy(f'{srcLabelDir}/{label}', f'{destLabelDir}/{label}')


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
    rmFromDirs([f'{dataName}/training/images',
               f'{dataName}/training/labels',
               f'{dataName}/validation/images',
               f'{dataName}/validation/labels',
               f'{dataName}/trainingGrey/images',
               f'{dataName}/trainingGrey/labels',
               f'{dataName}/validationGrey/images',
               f'{dataName}/validationGrey/labels'])
    
    
    
    subDirs = [f'{dataName}/training',
               f'{dataName}/validation',
               f'{dataName}/trainingGrey',
               f'{dataName}/validationGrey']
    
    for s in subDirs:
        shutil.rmtree(s, True)
    
    shutil.rmtree(dataName, True)

def buildTrainingFileStructure(dataName):
    os.mkdir(f'{dataName}')
    
    higherpaths = [f'{dataName}/training',
               f'{dataName}/validation',
               f'{dataName}/trainingGrey',
               f'{dataName}/validationGrey']
    for path in higherpaths:
        os.mkdir(path)
    
    paths = [f'{dataName}/training/images',
               f'{dataName}/training/labels',
               f'{dataName}/validation/images',
               f'{dataName}/validation/labels',
               f'{dataName}/trainingGrey/images',
               f'{dataName}/trainingGrey/labels',
               f'{dataName}/validationGrey/images',
               f'{dataName}/validationGrey/labels']
    for path in paths:
        os.mkdir(path)

def createYaml(yamlFile, dataName):
    f = open(yamlFile, 'w')
    
    f.write('path: ../photos\n')
    f.write(f'train: [existingdatasetsV2/fl_train/images, generatedData{dataName}/trainingGrey/images]\n')
    f.write(f'val: [existingdatasetsV2/fl_val/images, generatedData{dataName}/validationGrey/images]\n')
    f.write('nc: 1\nnames: [ \'drone\' ]\n')
      
    f.close()
    
def createYamlJustFL(yamlFile, dataName):
    f = open(yamlFile, 'w')
    
    f.write('path: ../photos\n')
    f.write(f'train: [existingdatasetsV2/fl_train/images]\n')
    f.write(f'val: [existingdatasetsV2/fl_val/images]\n')
    f.write('nc: 1\nnames: [ \'drone\' ]\n')
      
    f.close()
    
def createSlurm(slurm, dataName, yoloModel, yamlFile):
    f = open(slurm, 'w')
    
    f.write(bigText.slurmFirstPart)
    f.write(bigText.timeForYoloModel(yoloModel))
    f.write(bigText.jobName(dataName, yoloModel))
    f.write(bigText.slurmSecondPart)
    f.write(bigText.slurmArgs(yamlFile, yoloModel))
    f.write(bigText.slurmThirdPart)
      
    f.close()
    

def buildTrainingDataAndFiles(dataName):
    
    imgBaseFolder = f'photos/generatedData{dataName}'
    if os.path.isdir(imgBaseFolder):
        removeTrainingFileStructure(imgBaseFolder)
    buildTrainingFileStructure(imgBaseFolder)
    
    yamlFile = f'yamls/train_{dataName}.yaml'
    if os.path.isfile(yamlFile):
        os.remove(yamlFile)
    createYaml(yamlFile, dataName)
    
    buildSlurms(dataName, yamlFile)
        
    makeTrainingData('photos/drones/randrotate_5',
                     'photos/background/syntheticBgs/',
                     f'photos/generatedData{dataName}/training/images',
                     f'photos/generatedData{dataName}/training/labels')
    
    splitIntoTrainAndValidation(f'photos/generatedData{dataName}/training/images',
                                f'photos/generatedData{dataName}/validation/images')
    
    t1 = threading.Thread(target=makeGreyScaleCopy, args=(f'photos/generatedData{dataName}/training/images', 
                      f'photos/generatedData{dataName}/trainingGrey/images',
                      f'photos/generatedData{dataName}/training/labels',
                      f'photos/generatedData{dataName}/trainingGrey/labels') )
    t2 = threading.Thread(target=makeGreyScaleCopy, args= (f'photos/generatedData{dataName}/validation/images', 
                      f'photos/generatedData{dataName}/validationGrey/images',
                      f'photos/generatedData{dataName}/validation/labels',
                      f'photos/generatedData{dataName}/validationGrey/labels'))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
def buildSlurms(dataName, yamlFile):
    slurms = [f'slurms/train_{dataName}_n.slurm',
              f'slurms/train_{dataName}_s.slurm',
              f'slurms/train_{dataName}_m.slurm',
              f'slurms/train_{dataName}_l.slurm',
              f'slurms/train_{dataName}_x.slurm']
    yoloModels = ['yolov5n.pt',
                 'yolov5s.pt',
                 'yolov5m.pt',
                 'yolov5l.pt',
                 'yolov5x.pt',]
    for i, slurm in enumerate(slurms):
        if os.path.isfile(slurm):
            os.remove(slurm)
            print('removeing', slurm)
        print('creating slurm', slurm)
        createSlurm(slurm, dataName, yoloModels[i], yamlFile)
    
def rebuildData(dataName):
    
    imgBaseFolder = f'photos/generatedData{dataName}'
    if os.path.isdir(imgBaseFolder):
        removeTrainingFileStructure(imgBaseFolder)
    buildTrainingFileStructure(imgBaseFolder)
        
    makeTrainingData('photos/drones/trainAug',
                     bgTrainingDir,
                     f'photos/generatedData{dataName}/training/images',
                     f'photos/generatedData{dataName}/training/labels')
    
    splitIntoTrainAndValidation(f'photos/generatedData{dataName}/training/images',
                                f'photos/generatedData{dataName}/validation/images')
    
    t1 = threading.Thread(target=makeGreyScaleCopy, args=(f'photos/generatedData{dataName}/training/images', 
                      f'photos/generatedData{dataName}/trainingGrey/images',
                      f'photos/generatedData{dataName}/training/labels',
                      f'photos/generatedData{dataName}/trainingGrey/labels') )
    t2 = threading.Thread(target=makeGreyScaleCopy, args= (f'photos/generatedData{dataName}/validation/images', 
                      f'photos/generatedData{dataName}/validationGrey/images',
                      f'photos/generatedData{dataName}/validation/labels',
                      f'photos/generatedData{dataName}/validationGrey/labels'))
    
    t1.start()
    t2.start()
    
    t1.join()
    t2.join()
    
def buildTrainingDataAndFilesJustFL(dataName):
    
    yamlFile = f'yamls/train_{dataName}.yaml'
    if os.path.isfile(yamlFile):
        os.remove(yamlFile)
    createYamlJustFL(yamlFile, dataName)
    
    slurms = [f'slurms/train_{dataName}_n.slurm',
              f'slurms/train_{dataName}_s.slurm',
              f'slurms/train_{dataName}_m.slurm',
              f'slurms/train_{dataName}_l.slurm',
              f'slurms/train_{dataName}_x.slurm']
    yoloModels = ['yolov5n.pt',
                 'yolov5s.pt',
                 'yolov5m.pt',
                 'yolov5l.pt',
                 'yolov5x.pt',]
    for i, slurm in enumerate(slurms):
        if os.path.isfile(slurm):
            os.remove(slurm)
        createSlurm(slurm, dataName, yoloModels[i], yamlFile)
    

if __name__ == "__main__":
    rebuildData('fake')
    # buildSlurms('randrotate_5', 'yamls/train_randrotate_5.yaml')
    # buildSlurms('v2FL130', 'yamls/train_v2FL.yaml')
    
