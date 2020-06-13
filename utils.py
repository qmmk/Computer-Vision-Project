from matplotlib import pyplot as plt
import numpy as np
import cv2
import csv
import lensfunpy
from numpy import genfromtxt
import torch

lista_cvs = './dataset/data.csv'
feature_csv = './content/feature_vectors.csv'
map_frame = "./content/map.png"


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


def carica_feature_csv():
    feature_vectors = genfromtxt(feature_csv, delimiter=',')
    ret = torch.from_numpy(feature_vectors)
    return ret


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
    cv2.putText(frame, text, (x + 20, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return


def showImageAndStop(name, im):
    cv2.imshow(name, im)
    cv2.waitKey()
    cv2.destroyAllWindows()


def write_local(text, n_frame, n_quadro, warped):
    text_n = text.split('-')[0]
    path = "./rectifications/" + str(n_frame) + "_" + str(n_quadro) + "_" + text_n + ".jpg"
    cv2.imwrite(path, warped)
    n_quadro += 1
    return


cam_maker = 'GOPRO'
cam_model = 'HERO4 Silver'
lens_maker = 'GOPRO'
lens_model = 'fixed lens'

db = lensfunpy.Database()
cam = db.find_cameras(cam_maker, cam_model)[0]
lens = db.find_lenses(cam, lens_maker, lens_model)[0]

focal_length = 28.0
aperture = 1.4
distance = 10


def correct_distortion(frame, h, w):
    mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
    mod.initialize(focal_length, aperture, distance)

    undist_coords = mod.apply_geometry_distortion()
    im_undistorted = cv2.remap(frame, undist_coords, None, cv2.INTER_LANCZOS4)
    return im_undistorted


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def drawMap(stanza):
    map = cv2.imread(map_frame, cv2.IMREAD_COLOR)
    cv2.circle(map, drawPoint(stanza), 15, (0, 255, 0), -1)
    return map


def drawPoint(index):
    room = {
        "0": (584, 230),
        "1": (985, 514),
        "2": (985, 657),
        "3": (880, 657),
        "4": (773, 657),
        "5": (670, 657),
        "6": (563, 657),
        "7": (465, 657),
        "8": (386, 657),
        "9": (308, 657),
        "10": (236, 657),
        "11": (162, 657),
        "12": (57, 657),
        "13": (52, 512),
        "14": (59, 359),
        "15": (58, 218),
        "16": (78, 61),
        "17": (183, 61),
        "18": (268, 61),
        "19": (215, 258),
        "20": (260, 511),
        "21": (578, 514),
        "22": (827, 512)
    }
    return room.get(index)


def display(tmp, fh, fw, frame, roi, res):
    map = drawMap(tmp)

    display = np.zeros((fh * 3, fw * 3, 3), dtype="uint8")
    frame = cv2.resize(frame, (fw * 2 - 20, fh * 2 - 20))
    display[0 + 10:(fh * 2 - 10), 0 + 10:(fw * 2 - 10)] = frame
    roi = cv2.resize(roi, (roi.shape[1] - 20, roi.shape[0] - 20))
    display[0 + 10:fh - 10, fw * 2 + 10:fw * 3 - 10] = roi
    map = cv2.resize(map, (roi.shape[1], roi.shape[0]))
    display[fh + 10:fh * 2 - 10, fw * 2 + 10:fw * 3 + -10] = map

    for idx, r in enumerate(res):
        a = cv2.resize(res[idx]['before'], (roi.shape[1], roi.shape[0]))
        b = cv2.resize(res[idx]['after'], (roi.shape[1], roi.shape[0]))
        comb = np.hstack((a, b))
        comb = cv2.resize(comb, (roi.shape[1], roi.shape[0]))
        display[fh * 2 + 10:fh * 3 - 10, (fw * idx) + 10:(fw * (idx + 1)) - 10] = comb

    display = cv2.resize(display, (1080, 720))
    return display


def check_inside(text, rects, dict):
    index = []
    res = True
    for idx, d in enumerate(dict):
        # SE il nuovo quadro è contenuto in uno in dict che ha nome -quadro-
        # e il nuovo quadro ha nome -segnato-
        # ALLORA elimino il quadro in dict, ALTRIMENTI non aggiungo il nuovo quadro
        if isInside(rects, d['rects']) and d['texts'] == "quadro":
            if text != "quadro":
                index.append(idx)
            else:
                res = False

        # SE il nuovo quadro contiene uno in dict che ha nome -segnato-
        # e il nuovo quadro ha come nome -quadro-
        # ALLORA non lo aggiungo a dict, ALTRIMENTI tengo quello esterno
        if isOutside(rects, d['rects']) and text == "quadro":
            if d['texts'] != "quadro":
                res = False
            else:
                index.append(idx)

    for i in index:
        dict.pop(i)

    return res


def isInside(new, rects):
    x1, y1, w, h = rects
    x2, y2 = x1 + w, y1 + h
    X, Y, W, H = new
    if (x1 < X < x2) and (x1 < (X + W) < x2) and (y1 < Y < y2) and (y1 < (Y + H) < y2):
        return True
    return False


def isOutside(new, rects):
    x1, y1, w, h = new
    x2, y2 = x1 + w, y1 + h
    X, Y, W, H = rects
    if (x1 < X < x2) and (x1 < (X + W) < x2) and (y1 < Y < y2) and (y1 < (Y + H) < y2):
        return True
    return False