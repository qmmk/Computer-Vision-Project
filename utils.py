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
    # feature_vectors = []
    # with open(feature_csv, newline='') as csvfile:
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
                if (x1 < X and X < x2) and (x1 < (X + W) and (X + W) < x2):
                    if (y1 < Y and Y < y2) and (y1 < (Y + H) and (Y + H) < y2):
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
    if len(hulls) == 1:
        listindexfree = [0]
        return listindexfree
    listindexfree = contourIntersect(hulls, frame)
    listindexinside = checkInside(rects, listindexfree)
    print(listindexfree, listindexinside)
    listindexfree = set(listindexfree) - set(listindexinside)
    return listindexfree


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


def drawMap(map, stanza):
    stz = 0
    if stanza != "Stanza generica":
        try:
            stz = int(stanza)
        except ValueError:
            stz = 0
    cv2.circle(map, drawPoint(stz), 10, (0, 255, 0), -1)
    return map


def drawPoint(index):
    room = {
        0: (584, 230),
        1: (985, 514),
        2: (985, 657),
        3: (880, 657),
        4: (773, 657),
        5: (670, 657),
        6: (563, 657),
        7: (465, 657),
        8: (386, 657),
        9: (308, 657),
        10: (236, 657),
        11: (162, 657),
        12: (57, 657),
        13: (52, 512),
        14: (59, 359),
        15: (58, 218),
        16: (78, 61),
        17: (183, 61),
        18: (268, 61),
        19: (215, 258),
        20: (260, 511),
        21: (578, 514),
        22: (827, 512)
    }
    return room.get(index)


def display(room, res, frame, src_mask):
    map = cv2.imread(map_frame, cv2.IMREAD_COLOR)
    map = drawMap(map, room)
    vert = np.zeros(shape=(1, 600, 3))
    for dis in res:
        comb = np.hstack((cv2.resize(dis['not'], (300, 300)), cv2.resize(dis['yes'], (300, 300))))
        vert = np.vstack((vert, comb))

    fig, axes = plt.subplots(2, 2)
    axes[0][0].imshow(frame)
    axes[0][0].set_title('Detection')
    axes[0][1].imshow(src_mask)
    axes[0][1].set_title('Region of interest')
    axes[1][0].imshow(map)
    axes[1][0].set_title('Museum map')
    axes[1][1].imshow(comb)
    axes[1][1].set_title('Rectification')

    for ax in axes:
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['left'].set_visible(False)
        ax[0].spines['bottom'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].set_xticks([])
        ax[0].set_yticks([])

        ax[1].spines['top'].set_visible(False)
        ax[1].spines['left'].set_visible(False)
        ax[1].spines['bottom'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        ax[1].set_xticks([])
        ax[1].set_yticks([])

    plt.show()
    return
