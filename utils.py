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
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


def showImageAndStop(name, im):
    cv2.imshow(name, im)
    cv2.waitKey()
    cv2.destroyAllWindows()
