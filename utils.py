import numpy as np
import cv2
import csv

lista_cvs = './dataset/data.csv'


def carica_lista_cvs():
    lista_titoli = []
    lista_immagini = []
    lista_stanze = []
    with open(lista_cvs) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            lista_immagini.append(row[3])
            lista_titoli.append(row[0])
            lista_stanze.append(row[2])

    return lista_titoli, lista_immagini, lista_stanze


def compute_histogram(img):
    planes = []
    if len(img.shape) == 3:
        h, w, d = img.shape
        h_w = h * w
        if d == 3:
            p1 = img[:, :, 0]
            p2 = img[:, :, 1]
            p3 = img[:, :, 2]
            planes = [p1, p2, p3]
        else:
            planes = [img]

    if len(img.shape) == 2:
        h_w, d = img.shape
        if d == 3:
            p1 = img[:, 0]
            p2 = img[:, 1]
            p3 = img[:, 2]
            planes = [p1, p2, p3]
        else:
            planes = [img]

    histogram = np.zeros(256 * d)
    for i in np.arange(len(planes)):
        p = planes[i]
        for val in np.unique(p):
            count = np.sum(p == val)
            histogram[val + i * 256] = count
    histogram = histogram / img.size
    return histogram


def hist_compute_orb(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist


def entropy(histogram):
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))


def drawLabel(w, h, x, y, text, frame):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 0, 0), 2)
    cv2.putText(frame, text, (x+20, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


def showImageAndStop(name, im):
    cv2.imshow(name, im)
    cv2.waitKey()
    cv2.destroyAllWindows()


def contourIntersect(contours, frame):
    blank = np.zeros(frame.shape[0:2])
    conts = []
    intersection = []
    for i in range(len(contours)):
        for j in range(len(contours)):
            if i != j:
                checkcontours = [contours[i], contours[j]]
                # Copy each contour into its own image and fill it with '1'
                image1 = cv2.drawContours(blank.copy(), checkcontours, 0, 1)
                image2 = cv2.drawContours(blank.copy(), checkcontours, 1, 1)

                mat = cv2.bitwise_and(image1, image2)
                intersection.append(mat)

        for k in intersection:
            if k.any():
                intersection = []

        if len(intersection) != 0:
            conts.append(i)

    return conts


def checkInside(rects, index):
    new_index = []
    for i in index:
        for j in index:
            if i != j:
                x1, y1, w, h = rects[i]
                x2, y2 = x1 + w, y1 + h
                X, Y, W, H = rects[j]
                if (x1 < X and X < x2) and (x1 < (X+W) and (X+W) < x2):
                    if (y1 < Y and Y < y2) and (y1 < (Y+H) and (Y+H) < y2):
                        new_index.append(j)

    return new_index


def reduceListOuts(outs, rects, listindexfree):
    out_ = []
    rect_ = []
    for i in range(len(outs)):
        if i in listindexfree:
            out_.append(outs[i])
            rect_.append(rects[i])

    return out_, rect_


def shrinkenCountoursList(hulls, frame, rects):
    listindexfree = contourIntersect(hulls, frame)
    listindexinside = checkInside(rects, listindexfree)
    listindexfree = set(listindexfree) - set(listindexinside)
    return listindexfree
