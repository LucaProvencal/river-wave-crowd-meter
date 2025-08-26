import cv2
import numpy as np
from constants import YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH, YOLO_CLASSES_PATH, MATCH_CONFIDENCE

class Detector:
    def __init__(self):
        self.net = cv2.dnn.readNetFromDarknet(YOLO_CONFIG_PATH, YOLO_WEIGHTS_PATH)

        with open(YOLO_CLASSES_PATH, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def detect(self, image):
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0 / 255, size=(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        height, width, _ = image.shape

        boxes = []
        confidences = []

        for output in outputs:
            for detection in output:
                detection = np.array(detection)
                
                center_x = detection[0] * width
                center_y = detection[1] * height
                w = detection[2] * width
                h = detection[3] * height
                confidence = detection[5]
                
                # Tune this to allow more people matches.
                if confidence > MATCH_CONFIDENCE:
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))

        # Apply non-max suppression to filter overlapping boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.1, nms_threshold=0.1)
        boxes = [boxes[i] for i in indices]

        print(f"{len(boxes)} total people found")

        return boxes
