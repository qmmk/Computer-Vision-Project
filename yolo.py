import cv2
import utils
import numpy as np
import dlib

# Load Yolo
config = "./content/yolov3.cfg"
weights = "./content/yolov3.weights"
names = "./content/coco.names"
face_dat = "./content/shape_predictor_68_face_landmarks.dat"
net = cv2.dnn.readNet(weights, config)

# Face
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_dat)

with open(names, "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def detect_person(frame, height, width):
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

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            utils.drawLabel(w, h, x, y, label + " " + str(round(confidence, 2)), frame)
    return frame


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def detect_eyes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        dx = midpoint(landmarks.part(37), landmarks.part(40))
        sx = midpoint(landmarks.part(43), landmarks.part(46))

        cv2.circle(frame, dx, 5, (0, 255, 0), -1)
        cv2.circle(frame, sx, 5, (0, 255, 0), -1)

    return frame
