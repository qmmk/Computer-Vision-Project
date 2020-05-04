import numpy as np
import cv2
import csv
from PIL import Image

# input: img --> 2D or 3D array
# output: histogram normalized
lista_cvs = './dataset/data.csv'


def carica_lista_cvs():
    lista_titoli = []
    lista_immagini = []
    with open(lista_cvs) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            lista_immagini.append(row[3])
            lista_titoli.append(row[0])

    return lista_titoli, lista_immagini


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


# function for Shannon's Entropy
def entropy(histogram):
    histogram = histogram[histogram > 0]
    return -np.sum(histogram * np.log2(histogram))


kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)

g_kernel = cv2.getGaborKernel((25, 25), 6.5, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# g_kernel = cv2.getGaborKernel((30, 30), 6.5, np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_32F)
color = (255, 255, 255)


def hybrid_edge_detection(frame):
    gray_no_blur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray_no_blur, (5, 5), cv2.BORDER_DEFAULT)
    gray = cv2.GaussianBlur(gray, (15, 15), cv2.BORDER_DEFAULT)

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
    dilatation_out_canny = cv2.dilate(edges_canny, kernel2, iterations=5)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # erode = cv2.erode(thresh, kernel2, iterations=1)

    img_bwa = cv2.bitwise_and(thresh, dilatation_out)

    # img_bwa = cv2.bitwise_or(img_bwa, dilatation_out_canny)

    img_bwa = cv2.erode(img_bwa, kernel2, iterations=2)
    img_bwa = cv2.erode(img_bwa, kernel, iterations=7)
    img_bwa = cv2.dilate(img_bwa, kernel7, iterations=2)

    return thresh


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


def otsu(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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


"""'"""


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
        if m.distance < 50:  # 40
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
        print("{} -> score: {}".format(titolo_immagine, score))

        if score < 250:
            print(score)
            img3 = cv2.drawMatches(im1, kp1, im2, kp2, good[:ngood], None, flags=2)
            cv2.imshow(titolo_immagine, img3)
            cv2.waitKey()
            cv2.destroyAllWindows()
            return True, good, retkp1, retkp2, score
        else:
            return False, 0, 0, 0, 100000
    else:
        return False, 0, 0, 0, 100000


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
