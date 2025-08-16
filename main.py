import cv2
import numpy as np
import math

# Paths to the YOLOv3 model files
config_path = 'yolov3.cfg'
weights_path = 'yolov3.weights'
class_names_path = 'coco.names'

# Load YOLO
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# Load class names
with open(class_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load an image
image = cv2.imread('assets/17.PNG')
height, width, _ = image.shape

# Prepare the image for the network
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0 / 255, size=(416, 416), swapRB=True, crop=False)
net.setInput(blob)

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Perform detection
outputs = net.forward(output_layers)

# Extracting the bounding boxes, confidences, and class ids
boxes = []
confidences = []
class_ids = []

# Process the detections
for output in outputs:
    for detection in output:
        # Ensure detection is a numpy array and has 85 values
        detection = np.array(detection)
        
        # Extract values
        center_x = detection[0] * width
        center_y = detection[1] * height
        w = detection[2] * width
        h = detection[3] * height
        objectness = detection[4]
        confidence = detection[5]
        
        # Only consider detections with high confidence
        if confidence > 0.5:
            print(objectness)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(5)

# Apply non-max suppression to filter overlapping boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.6)

# Draw bounding boxes and labels
# for i in range(len(boxes)):

for i in indices:
    box = boxes[i]
    x, y, w, h = box

    cv2.rectangle(image, (x, y), (math.floor(x + w), math.floor(y + h)), (0, 255, 0), 2)

# Display the image with detections
cv2.imshow('Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
