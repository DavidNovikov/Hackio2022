import cv2
import numpy as np
import torch

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)
# model_path = 'best_3.pt'
# model_conf = 0.1
# isSahi = True
# # isSahi = False
# model = torch.hub.load(
#     '../yolov5', 'custom', path=model_path, source='local')
# model.conf = model_conf
# model.iou = 0.45

# def drawboxWithNameAndConf(result, image):
#     h, w, _ = image.shape
#     x1 = int(result.xmin*w)
#     y1 = int(result.ymin*h)
#     x2 = int(result.xmax*w)
#     y2 = int(result.ymax*h)
#     cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
#     cv2.putText(image, f'{result.values[6]} {result.confidence:.2f}',
#                 (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

def video():
    video = cv2.VideoCapture('vid1.mp4')
    ret, frame = video.read()
    h, w, _ = frame.shape
    while ret:
        frame = cv2.resize(frame, (640, 640), cv2.INTER_AREA)
        frameForModel = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # pre processing
        # results = inference_with_sahi(frameForModel)
        results = model(frameForModel)

        d = False
        
        detections = []
        sizes = []

        for _, r in results.pandas().xyxyn[0].iterrows():
            if r.values[6] == 'person' and r.confidence > 0.4:
                x1 = int(r.xmin*w)
                y1 = int(r.ymin*h)
                x2 = int(r.xmax*w)
                y2 = int(r.ymax*h)
                detections.append([x1, y1, x2, y2])
                sizes.append((x2-x1)*(y2-y1))
                d = True
        
        if not d:
            print('failed to find')
        else:
            maxI = np.argmax(sizes)
            
            detection = detections[maxI]
            
            x1, y1, x2, y2 = detection
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f'person ',
                        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
                
        
        

        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        ret, frame = video.read()


# images()
video()