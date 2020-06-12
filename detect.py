import cv2
import numpy as np
import utils
from torch.autograd import Variable
import torch

lista_titoli, lista_immagini, lista_stanze = utils.carica_lista_cvs()

kernel = np.ones((3, 3), np.uint8)
kernel1 = np.ones((1, 1), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)
kernel7 = np.ones((7, 7), np.uint8)

g_kernel = cv2.getGaborKernel((25, 25), 6.5, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# g_kernel = cv2.getGaborKernel((30, 30), 6.5, np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_32F)

color = (255, 255, 255)


def hybrid_edge_detection_V2(frame, no_gabor=False):
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


def get_contours(src):
    _, conts, heirarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # CHAIN_APPROX_NONE

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

    return rects, hull_list, src_mask


def cropping_frame(frame, hulls, src_mask):
    outs = []
    masks = []
    green = []

    # loop per estrarre e appendere a liste predifinite crop immagini
    for i in range(len(hulls)):
        outs.append(image_crop(frame, hulls, i, 0))
        masks.append(image_crop(src_mask, hulls, i, 1))
        green.append(image_crop(frame, hulls, i, 2))
    return outs, masks, green


def image_crop(frame, hull_list, i, param):
    mask = np.zeros_like(frame)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, hull_list, i, color, -1)  # Draw filled contour in mask
    out = np.zeros_like(frame)  # Extract out the object and place into output image
    if param == 2:
        out[:, :, 1] = 255
    out[mask == 255] = frame[mask == 255]

    # Now crop
    if param == 1:
        (y, x) = np.where(mask == 255)
    else:
        (y, x, z) = np.where(mask == 255)

    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy + 1, topx:bottomx + 1]
    return out


def get_feature_vector(img, scaler, to_tensor, normalize, layer, resnet):
    #  Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    #  Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    # my_embedding = torch.zeros(512)
    my_embedding = torch.zeros([1, 512, 1, 512])

    #  Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)

    #  Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    #  Run the model on our transformed image
    resnet(t_img)
    #  Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding
