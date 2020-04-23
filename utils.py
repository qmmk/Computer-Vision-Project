import numpy as np
import cv2


# input: img --> 2D or 3D array
# output: histogram normalized
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
g_kernel = cv2.getGaborKernel((25, 25), 6.5, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)


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
