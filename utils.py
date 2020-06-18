import numpy as np
import cv2
import csv
import lensfunpy
from numpy import genfromtxt
import torch
from scipy import ndimage

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
    l = 0
    testo = text.split()
    fin = ''
    for letter in testo:
        if l == 4:
            fin += '\n'
            l = 0
        l += 1
        fin = fin + letter + " "

    for i, line in enumerate(fin.split('\n')):
        y = y + 20
        cv2.putText(frame, line, (x + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return


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
    roi = cv2.resize(roi, (fw - 20, fh - 20))
    display[0 + 10:fh - 10, (fw * 2 + 10):(fw * 3 - 10)] = roi
    map = cv2.resize(map, (fw - 20, fh - 20))
    display[fh + 10:(fh * 2 - 10), (fw * 2 + 10):(fw * 3 + -10)] = map

    for idx, r in enumerate(res):
        a = r['before']
        b = r['after']
        a, b = stack(a, b)
        comb = np.hstack((a, b))

        comb = image_resize(comb, width=fw - 20)
        h, w, _ = comb.shape

        fxh = (fh - 20) / h
        fxw = (fw - 20) / w

        if fxw > fxh:
            comb = cv2.resize(comb, None, fx=fxh, fy=fxh)
            h, w, _ = comb.shape
            t = ((fw - 20) - w) % 2
            comb = cv2.copyMakeBorder(comb, 0, 0, int(((fw - 20) - w) / 2), int(((fw - 20) - w) / 2) + int(t), 0)

        else:
            comb = cv2.resize(comb, None, fx=fxw, fy=fxw)
            h, w, _ = comb.shape
            t = ((fh - 20) - h) % 2
            comb = cv2.copyMakeBorder(comb, int(((fh - 20) - h) / 2), int(((fh - 20) - h) / 2) + int(t), 0, 0, 0)

        display[fh * 2 + 10:fh * 3 - 10, (fw * idx) + 10:(fw * (idx + 1)) - 10] = comb

    display = cv2.resize(display, (1920, 1080))
    return display


def stack(img1, img2):
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape

    if w1 > w2:
        t = (w1 - w2) % 2
        img2 = cv2.copyMakeBorder(img2, 0, 0, int((w1 - w2) / 2), int((w1 - w2) / 2) + t, 0)
    if w1 < w2:
        t = (w2 - w1) % 2
        img1 = cv2.copyMakeBorder(img1, 0, 0, int((w2 - w1) / 2), int((w2 - w1) / 2) + t, 0)
    if h1 > h2:
        t = (h1 - h2) % 2
        img2 = cv2.copyMakeBorder(img2, int((h1 - h2) / 2), int((h1 - h2) / 2) + t, 0, 0, 0)
    if h1 < h2:
        t = (h2 - h1) % 2
        img1 = cv2.copyMakeBorder(img1, int((h2 - h1) / 2), int((h2 - h1) / 2) + t, 0, 0, 0)

    return img1, img2


def check_inside(text, rects, dict):
    index = []
    res = True
    for idx, d in enumerate(dict):
        '''
        SE il nuovo quadro è contenuto in uno in dict che nome -quadro-
            SE il nuovo quadro ha nome -segnato-, ALLORA rimuove quello in dict
            ALTRIMENTI non aggiungo il nuovo quadro
        '''
        if isInside(rects, d['rects']) and d['texts'] == "quadro":
            if text != "quadro":
                index.append(idx)
            else:
                res = False

        '''
        SE il nuovo quadro con nome -quadro- contiene uno in dict 
            SE il quadro in dict ha nome -segnato-, ALLORA non aggiungo il nuovo
            ALTRIMENTI rimuove quello in dict
        '''
        if isOutside(rects, d['rects']) and text == "quadro":
            if d['texts'] != "quadro":
                res = False
            else:
                index.append(idx)

        '''
        SE il nuovo quadro e quello in dict hanno nome -segnato-
            SE il nuovo quadro è contenuto in uno in dict ed ha score minore, OPPURE 
                il nuovo quadro contiene uno in dict ed ha score minore    
            ALLORA tolgo quello in dict
            
            ALTRIMENTI non aggiungo il nuovo
        '''
        if text != "quadro" and d['texts'] != "quadro":
            score1 = [int(s) for s in text.split() if s.isdigit()]
            score2 = [int(s) for s in d['texts'].split() if s.isdigit()]
            if (isInside(rects, d['rects']) and score1[0] <= score2[0]) or \
                    (isOutside(rects, d['rects']) and score1[0] <= score2[0]):
                index.append(idx)
            else:
                res = False

    for i in reversed(index):
        dict.pop(i)

    return res


def isInside(new, old):
    x1, y1, w, h = old
    x2, y2 = x1 + w, y1 + h
    X, Y, W, H = new
    if (x1 < X < x2) and (x1 < (X + W) < x2) and (y1 < Y < y2) and (y1 < (Y + H) < y2):
        return True
    return False


def isOutside(new, old):
    x1, y1, w, h = new
    x2, y2 = x1 + w, y1 + h
    X, Y, W, H = old
    if (x1 < X < x2) and (x1 < (X + W) < x2) and (y1 < Y < y2) and (y1 < (Y + H) < y2):
        return True
    return False


def rotate(frame):
    # rotation angle in degree
    frame = ndimage.rotate(frame, 270)
    return frame


def resize_output(frame):
    H = 1080
    W = 1920
    frame_width = frame.shape[0]
    frame_height = frame.shape[1]
    fxh = H / frame_width
    fxw = W / frame_height

    if fxh > 1 and fxw > 1:
        fxh = frame_width / H
        fxw = frame_height / W

    if fxw > fxh:

        frame = cv2.resize(frame, None, fx=fxh, fy=fxh)
        h1, w1, _ = frame.shape
        t = (W - w1) % 2
        frame = cv2.copyMakeBorder(frame, 0, 0, int((W - w1) / 2), int((W - w1) / 2) + int(t), 0)

    else:
        if fxh == 1 and fxw == 1 and rotate:
            frame = cv2.resize(frame, None, fx=fxh, fy=fxh)
            w1, h1, _ = frame.shape
            t = (W - w1) % 2
            frame = cv2.copyMakeBorder(frame, 0, 0, int((W - w1) / 2), int((W - w1) / 2) + int(t), 0)

        else:
            frame = cv2.resize(frame, None, fx=fxw, fy=fxw)
            h1, w1, _ = frame.shape
            t = (H - h1) % 2
            frame = cv2.copyMakeBorder(frame, int((H - h1) / 2), int((H - h1) / 2) + int(t), 0, 0, 0)
    return frame
