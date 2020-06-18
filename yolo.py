import cv2
import numpy as np
import ffmpeg

# Load Yolo
config = "./content/yolov3.cfg"
weights = "./content/yolov3.weights"
names = "./content/coco.names"
net = cv2.dnn.readNet(weights, config)

with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detect_person(frame, height, width, dict):
    rects = []

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append((x, y, w, h))
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            if check_inside(boxes[i], dict):
                rects.append({'texts': "person " + str(round(confidences[i], 2)), 'rects': boxes[i]})
    dict.extend(rects)
    return dict


def check_inside(person, rects):
    for r in rects:
        x1, y1, w, h = r['rects']
        x2, y2 = x1 + w, y1 + h
        X, Y, W, H = person
        if (x1 <= X <= x2) and (x1 <= (X + W) <= x2) and (y1 <= Y <= y2) and (y1 <= (Y + H) <= y2):
            return False
    return True
