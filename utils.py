import numpy as np
import cv2
import csv
import scipy.spatial.distance
import math
from PIL import Image

# input: img --> 2D or 3D array
# output: histogram normalized
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

    # e' corretto 256, non h_w
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

# function for Shannon's Entropy
def entropy(histogram):
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))


kernel = np.ones((3, 3), np.uint8)
kernel1 = np.ones((1, 1), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)

g_kernel = cv2.getGaborKernel((25, 25), 6.5, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
#g_kernel = cv2.getGaborKernel((30, 30), 6.5, np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_32F)
color = (255, 255, 255)


def hybrid_edge_detection(frame):
    gray_no_blur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray_no_blur, (5, 5), cv2.BORDER_DEFAULT)
    gray = cv2.GaussianBlur(gray, (13, 13), cv2.BORDER_DEFAULT)

    gabor = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)

    edges_canny = cv2.Canny(gray_no_blur, 100, 200)

    # adaptive
    # edges = cv2.adaptiveThreshold(gabor, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.bitwise_not(gabor)

    # morpho
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    dilatation_out = cv2.dilate(opening, kernel7, iterations=3)

    # morpho_canny
    # opening = cv2.morphologyEx(edges_canny, cv2.MORPH_OPEN, kernel)
    dilatation_out_canny = cv2.dilate(edges_canny, kernel7, iterations=2)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # erode = cv2.erode(thresh, kernel2, iterations=1)

    img_bwa = cv2.bitwise_and(thresh, dilatation_out)

    img_bwa = cv2.erode(img_bwa, kernel2, iterations=2)
    img_bwa = cv2.erode(img_bwa, kernel, iterations=7)
    img_bwa_ = cv2.dilate(img_bwa, kernel7, iterations=2)

    img_bwa = cv2.bitwise_or(img_bwa_, dilatation_out_canny)

    """
    gabor = cv2.dilate(gabor,kernel2,iterations=1)
    img_bwa = cv2.bitwise_and(img_bwa_, gabor)
    """

    ad = adaptive_Filter(frame)
    img_bwa = cv2.bitwise_or(img_bwa, ad)

    img_bwa = cv2.erode(img_bwa, kernel, iterations=1)

    # cv2.imshow('hybrid',img_bwa)
    # cv2.waitKey()

    return img_bwa


class ColourBounds:
    def __init__(self, rgb):
        hsv = cv2.cvtColor(np.uint8([[[rgb[2], rgb[1], rgb[0]]]]), cv2.COLOR_BGR2HSV).flatten()

        lower = [hsv[0] - 10]
        upper = [hsv[0] + 10]

        if lower[0] < 0:
            lower.append(179 + lower[0])  # + negative = - abs
            upper.append(179)
            lower[0] = 0
        elif upper[0] > 179:
            lower.append(0)
            upper.append(upper[0] - 179)
            upper[0] = 179

        self.lower = [np.array([h, 100, 100]) for h in lower]
        self.upper = [np.array([h, 255, 255]) for h in upper]

colourMap = {
    "quadro": ColourBounds((150, 130, 100))
}


def adaptive(frame):
    for name, colour in colourMap.items():
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, colour.lower[0], colour.upper[0])

        if len(colour.lower) == 2:
            mask = mask | cv2.inRange(hsv, colour.lower[1], colour.upper[1])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # g_kernel = cv2.getGaborKernel((15, 15), 6.5, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        # #se usi questo kernel per entrambi è più stabile ma non prende quadro sbiadito

        g_kernel = cv2.getGaborKernel((15, 15), 8.0, np.pi / 4, 10.0, 0.5, 0.5, ktype=cv2.CV_32F)
        g_kernel2 = cv2.getGaborKernel((15, 15), 8.5, np.pi / 4, 10, 0.5, 0, ktype=cv2.CV_32F)
        gray = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
        gray = cv2.GaussianBlur(gray, (7, 7), 15)
        gray = cv2.GaussianBlur(gray, (7, 7), 15)
        gray = cv2.GaussianBlur(gray, (7, 7), 15)

        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        edges = cv2.bitwise_not(edges)
        erosion = cv2.erode(edges, kernel, iterations=2)
        erosion = cv2.medianBlur(erosion, 3)
        erosion_f = cv2.filter2D(erosion, cv2.CV_8UC3, g_kernel2)
        dilatation_out = cv2.dilate(erosion_f, kernel2, iterations=7)
        erosion2 = cv2.erode(dilatation_out, kernel2, iterations=2)
        scr_dilat = [erosion2.copy()]
    return scr_dilat

def hybrid_edge_detection_V2(frame):
    gray_no_blur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray_no_blur, (5, 5), cv2.BORDER_DEFAULT)
    gray = cv2.GaussianBlur(gray, (13, 13), cv2.BORDER_DEFAULT)

    gabor = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
    gabor = cv2.bitwise_not(gabor)

    dilate_gabor = cv2.dilate(gabor, kernel2, iterations=2)

    adapt_filter = adaptive_Filter(frame)

    canny = cv2.Canny(gray_no_blur, 50, 140)
    dilate_canny = cv2.dilate(canny, kernel2, iterations=1)

    img_bwa = cv2.bitwise_and(adapt_filter, dilate_canny)
    img_bwa = cv2.bitwise_or(img_bwa, dilate_gabor)

    #img_bwa = cv2.erode(img_bwa, kernel2, iterations=2)
    #img_bwa = cv2.erode(img_bwa, kernel, iterations=7)
    img_bwa = cv2.dilate(img_bwa, kernel, iterations=3)

    img_bwa = cv2.bitwise_or(adapt_filter, img_bwa)

    #showImageAndStop('f',img_bwa)

    return img_bwa

def adaptive_Filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 15)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.bitwise_not(edges)
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(opening, kernel2, iterations=2)

    return dilate

def otsu(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    erode = cv2.erode(thresh, kernel2, iterations=1)
    return erode

def drawLabel(w, h, x, y, text, frame):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 0, 0), 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

def image_crop(frame, hull_list, i):
    outs = []
    mask = np.zeros_like(frame)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, hull_list, i, color, -1)  # Draw filled contour in mask
    out = np.zeros_like(frame)  # Extract out the object and place into output image
    out[mask == 255] = frame[mask == 255]

    # Now crop
    (y, x, z) = np.where(mask == 255)
    # (y, x) = np.where(mask == 255)

    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy + 1, topx:bottomx + 1]
    # outs.append(out)
    return out

def image_crop_bin(frame, hull_list, i):
    outs = []
    mask = np.zeros_like(frame)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, hull_list, i, color, -1)  # Draw filled contour in mask
    out = np.zeros_like(frame)  # Extract out the object and place into output image
    out[mask == 255] = frame[mask == 255]

    # Now crop
    (y, x) = np.where(mask == 255)

    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy + 1, topx:bottomx + 1]
    outs.append(out)
    return outs

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def rectify_image(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually

    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def rectify_image_with_correspondences(im, p1, p2, w, h):
    m, status = cv2.findHomography(p1, p2)
    warped = cv2.warpPerspective(im, m, (w, h))

    return warped

def getLine(edges, frame):
    # get contours

    minLineLength = 100

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)
    if lines is None:
        return

    N = lines.shape[0]
    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]

        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    '''
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    '''

def ORB(im1, im2, titolo_immagine):
    # Initiate SIFT detector
    orb = cv2.ORB_create()
    # cv2.imshow("im1", im1)
    # cv2.imshow("im2", im2)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    if (des1 is None) or (des2 is None):
        return False, 0, 0, 0, 100000

    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    good = []
    retkp1 = []
    retkp2 = []
    ngood = 10

    for m in matches:
        if m.distance < 40:  # 50
            good.append(m)
            # Get the matching keypoints for each of the images
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            retkp1.append((x1, y1))
            retkp2.append((x2, y2))

    if len(good) >= ngood:
        good = sorted(good, key=lambda x: x.distance)
        score = sum(x.distance for x in good[:ngood])
        #print("{} -> score: {}".format(titolo_immagine, score))
        if score < 350: #230
            img3 = cv2.drawMatches(im1, kp1, im2, kp2, good[:ngood], None, flags=2)
            cv2.imshow(titolo_immagine, img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
            return True, good, retkp1, retkp2, score
        else:
            return False, 0, 0, 0, 100000
    else:
        return False, 0, 0, 0, 100000

def detectKeyPoints(lista_immagini,lista_titoli, lista_stanze, img_rgb):
    min_idx = -1
    min_score = 100000
    text = "quadro"
    room = ""
    for it in range(len(lista_immagini) - 1):
        # Read the main image
        titolo_quadro = lista_titoli[it + 1]
        immage_template = "./template/" + lista_immagini[it + 1]
        stanza = lista_stanze[it + 1]
        template = cv2.imread(immage_template,1)

        is_detected, matches, ret_kp1, ret_kp2, score = ORB(img_rgb, template, titolo_quadro)
        if score < min_score:
            min_score = score
            text = "{} - score: {}".format(titolo_quadro, score)
            room = "Stanza n.{}".format(stanza)

            min_idx = it
            array1 = np.array((ret_kp1), dtype=np.float32)
            array2 = np.array((ret_kp2), dtype=np.float32)

    if min_score < 100000:
        id = min_idx
        print("idx" + str(id))
        warped = rectify_image_with_correspondences(img_rgb, array2[:8], array1[:8], 1000, 1000)
        showImageAndStop(text,warped)

    return text, room


def fucking_yolo(frame, height, width):
    # Load Yolo
    config = "./content/yolov3.cfg"
    weights = "./content/yolov3.weights"
    names = "./content/coco.names"
    net = cv2.dnn.readNet(weights, config)
    font = cv2.FONT_HERSHEY_PLAIN
    classes = []
    with open(names, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (128, 128), (0, 0, 0), True, crop=False)

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
            color = colors[class_ids[i]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
    return frame

def hougesLinesAndCorner(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 100, np.array([]), 0, 0)
    out_line = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(out_line, (x1, y1), (x2, y2), (255, 255, 255), 1)

    #img_gray = cv2.cvtColor(out_line, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('out', out_line)
    #cv2.waitKey()

    corners = cv2.goodFeaturesToTrack(out_line, 4, 0.4, 80)

    if corners is not None:
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(out_line, (x, y), 3, 255, -1)

    else:
        corners = []
        return corners

    #showImageAndStop('hough', out_line)

    return corners

def showImageAndStop(name,im):
    cv2.imshow(name,im)
    cv2.waitKey()
    cv2.destroyAllWindows()

def order_corners(corners):
    p = []
    #p.append((corners[2][0][0], corners[2][0][1]))
   # p.append((corners[3][0][0], corners[3][0][1]))
   # p.append((corners[1][0][0], corners[1][0][1]))
   # p.append((corners[0][0][0], corners[0][0][1]))
    sumx = corners[2][0][0]+corners[3][0][0]+corners[1][0][0]+corners[0][0][0]
    sumy = corners[2][0][1] + corners[3][0][1] + corners[1][0][1] + corners[0][0][1]
    medx = sumx / 4
    medy = sumy / 4

    for c in corners[:]:
        #bottom left
        if c[0][0] < medx and c[0][1] < medy:
            bl = (c[0][0],c[0][1])
        # bottom right
        if c[0][0] > medx and c[0][1] < medy:
            br = (c[0][0], c[0][1])
        #top left
        if c[0][0] < medx and c[0][1] > medy:
            tl = (c[0][0], c[0][1])
        #top right
        if c[0][0] > medx and c[0][1] > medy:
            tr = (c[0][0], c[0][1])
    try:
        p.append(bl)
        p.append(br)
        p.append(tl)
        p.append(tr)
    except UnboundLocalError:
        print("The corner are wrong")
        return 0

    return p


def rectify_image_2(rows, cols, img, p):
    # image center
    u0 = (cols) / 2.0
    v0 = (rows) / 2.0
    # widths and heights of the projected image
    w1 = scipy.spatial.distance.euclidean(p[0], p[1])
    w2 = scipy.spatial.distance.euclidean(p[2], p[3])

    h1 = scipy.spatial.distance.euclidean(p[0], p[2])
    h2 = scipy.spatial.distance.euclidean(p[1], p[3])

    w = max(w1, w2)
    h = max(h1, h2)

    # visible aspect ratio
    ar_vis = float(w) / float(h)

    # make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0], p[0][1], 1)).astype('float32')
    m2 = np.array((p[1][0], p[1][1], 1)).astype('float32')
    m3 = np.array((p[2][0], p[2][1], 1)).astype('float32')
    m4 = np.array((p[3][0], p[3][1], 1)).astype('float32')

    # calculate the focal disrance
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    #per evitare divisioni per 0
    try:
        f = math.sqrt(np.abs((1.0 / (n23 * n33)) * ((n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) + (n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))))

        A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype('float32')

        At = np.transpose(A)
        Ati = np.linalg.inv(At)
        Ai = np.linalg.inv(A)

        # calculate the real aspect ratio
        ar_real = math.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))

        if ar_real < ar_vis:
            W = int(w)
            H = int(W / ar_real)
        else:
            H = int(h)
            W = int(ar_real * H)

        pts1 = np.array(p).astype('float32')
        pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])

        # project the image with the new w/h
        M = cv2.getPerspectiveTransform(pts1, pts2)

        dst = cv2.warpPerspective(img, M, (W, H))

        # cv2.imshow('img', img)
        # cv2.imshow('dst', dst)

        # cv2.waitKey(0)
    except ValueError:
        print("error in the formula")
        return 0

    return dst