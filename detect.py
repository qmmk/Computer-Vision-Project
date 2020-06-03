import cv2
import numpy as np
import utils

import torch
import torchvision.models as models
from torchvision import transforms
from torch.autograd import Variable
import csv
from PIL import Image
import torch.nn as nn
import torch

lista_titoli, lista_immagini, lista_stanze = utils.carica_lista_cvs()

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

kernel = np.ones((3, 3), np.uint8)
kernel1 = np.ones((1, 1), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)

g_kernel = cv2.getGaborKernel((25, 25), 6.5, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# g_kernel = cv2.getGaborKernel((30, 30), 6.5, np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_32F)

color = (255, 255, 255)

def optionalFilter(img):

    rgb_planes = cv2.split(img)

    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result_norm = cv2.merge(result_norm_planes)
    # result = cv2.merge(result_planes)
    result_norm = cv2.bitwise_not(result_norm)

    result_norm = cv2.Canny(result_norm, 50, 140)
    result_norm = cv2.dilate(result_norm, kernel2, iterations=1)

    utils.showImageAndStop('nosh',result_norm)


    return result_norm


def hybrid_edge_detection_V2(frame,no_gabor=False):
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


    if no_gabor == True:
        img_bwa = cv2.bitwise_or(img_bwa, dilate_gabor)


    # img_bwa = cv2.erode(img_bwa, kernel2, iterations=2)
    img_bwa = cv2.erode(img_bwa, kernel, iterations=3)
    img_bwa = cv2.dilate(img_bwa, kernel, iterations=5)


    img_bwa = cv2.bitwise_or(adapt_filter, img_bwa)

    img_bwa = np.where(img_bwa == 0, img_bwa, 255)

    return img_bwa


def adaptive_Filter(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 15)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    edges = cv2.bitwise_not(edges)
    opening = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    dilate = cv2.dilate(opening, kernel2, iterations=2)

    return dilate


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
    gray = cv2.GaussianBlur(gray, (5, 5), cv2.BORDER_DEFAULT)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    erode = cv2.erode(thresh, kernel2, iterations=1)
    return erode


def get_contours(src):

    _ ,conts, heirarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  #CHAIN_APPROX_NONE

    src_mask = np.zeros_like(src)
    hull_list = []
    rects = []
    for i in conts:
        rect = cv2.boundingRect(i)
        x, y, w, h = rect
        if w > 100 and h > 100:
            hull = cv2.convexHull(i)
            hull_list.append(hull)
            # creo contorni da dare alla funzione getline e un frame nero
            cv2.drawContours(src_mask, [hull], 0, (255, 255, 255), 1)
            rects.append(rect)
    #utils.showImageAndStop('ROI with intersection',src_mask)
    return rects, hull_list, src_mask


def cropping_frame(frame, hulls, src_mask):
    outs = []
    masks = []
    green = []

    # loop per estrarre e appendere a liste predifinite crop immagini
    for i in range(len(hulls)):
        outs.append(image_crop(frame, hulls, i))
        masks.append(image_crop_bin(src_mask, hulls, i))
        green.append(image_crop_green(frame, hulls, i))
    return outs, masks, green


def image_crop(frame, hull_list, i):
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
    return out

def image_crop_green(frame, hull_list, i):
    mask = np.zeros_like(frame)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, hull_list, i, color, -1)  # Draw filled contour in mask
    out = np.zeros_like(frame)  # Extract out the object and place into output image
    out[:, :, 1] = 255
    out[mask == 255] = frame[mask == 255]

    # Now crop
    (y, x, z) = np.where(mask == 255)
    # (y, x) = np.where(mask == 255)

    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy + 1, topx:bottomx + 1]
    return out


def image_crop_bin(frame, hull_list, i):
    mask = np.zeros_like(frame)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, hull_list, i, color, -1)  # Draw filled contour in mask
    out = np.zeros_like(frame)  # Extract out the object and place into output image
    out[mask == 255] = frame[mask == 255]

    # Now crop
    (y, x) = np.where(mask == 255)

    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy + 1, topx:bottomx + 1]
    return out
