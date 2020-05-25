from wand.image import Image
import numpy as np
import cv2
import csv
import lensfunpy

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
    listindexfree = contourIntersect(hulls, frame)
    listindexinside = checkInside(rects, listindexfree)
    listindexfree = set(listindexfree) - set(listindexinside)
    return listindexfree


cam_maker = 'GOPRO'
cam_model = 'HERO4 Silver'
lens_maker = 'GOPRO'
lens_model = 'fixed lens'

db = lensfunpy.Database()
print(db.find_cameras(cam_maker, cam_model)[0])
cam = db.find_cameras(cam_maker, cam_model)[0]
lens = db.find_lenses(cam, lens_maker, lens_model)[0]

focal_length = 28.0
aperture = 1.4
distance = 10
def correct_distortion(frame, h, w):
    # Definisci matrice telecamera K
    '''
    K = np.array([[[673.9683892, 0., 343.68638231],
                   [0., 676.08466459, 245.31865398],
                   [0., 0., 1.]]])

    # Definisce i coefficienti di distorsione d
    #d = np.array([5.44787247e-02, 1.23043244e-01, - 4.52559581e-04, 5.47011732e-03, - 6.83110234e-01])
    d = np.array([0.3, 0.001, 0.0, 0.0, 0.01])

    # Leggi un'immagine di esempio e acquisisci le sue dimensioni
    # img = cv2.imread("calibrazione_campioni / 2016-07-13-124020.jpg")
    # h, w = img.shape[: 2]

    # Genera nuova matrice telecamera dai parametri
    # newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 0)

    # Genera tabelle di ricerca per rimappare l' immagine della telecamera

    # mapx, mapy = cv2.initUndistortRectifyMap(K, d, None, newcameramatrix, (w, h), 5)

    # Rimappa l'immagine originale in una nuova immagine
    # newimg = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

    with Image(frame) as img:
        #print(img.size)
        img.virtual_pixel = 'transparent'
        img.distort('barrel', -(0.2, 0.0, 0.0, 1.0))
        # img.save(filename='checks_barrel.png')
        # convert to opencv/numpy array format
        img_opencv = np.array(img)
    '''


    #image_path = '/path/to/image.tiff'
    #undistorted_image_path = '/path/to/image_undist.tiff'

    #im = cv2.imread(image_path)
    #height, width = im.shape[0], im.shape[1]

    mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
    mod.initialize(focal_length, aperture, distance)

    undist_coords = mod.apply_geometry_distortion()
    im_undistorted = cv2.remap(frame, undist_coords, None, cv2.INTER_LANCZOS4)
    return im_undistorted
