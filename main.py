import cv2
import numpy as np
import utils
import time
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

video_stronzo = './videos/VIRB0391.MP4'
video_normale = './videos/VIRB0414.MP4'
video_tondo = './videos/GOPR2051.MP4'
video_comune = './videos/VIRB0407.MP4'
video_madonna_bimbo = './videos/VIRB0392.MP4'
video_boh = "./videos/VID_20180529_112440.mp4" #ci vuole gabor filter da 30
video_comune_trim = "./videos/trim.mp4"
video_trim = "./videos/trimsss.mp4"
video_diverso = './videos/VIRB0415.MP4'
video_gente = './videos/GOPR1940.MP4'
video_1 = './videos/GOPR1928.MP4'
video_2 = './videos/GOPR1947.MP4'
video_3 = './videos/GOPR2039.MP4'
video_nome_lungo = './videos/VID_20180529_112539.mp4'

cap = cv2.VideoCapture(video_3)

if (cap.isOpened() == False):
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

# global used variable
color = (255, 255, 255)
kernel = np.ones((3,3),np.uint8)
kernel2 = np.ones((5,5),np.uint8)
kernel3 = np.ones((7,7),np.uint8)

lista_titoli, lista_immagini, lista_stanze = utils.carica_lista_cvs()
room = "Stanza generica"

im = cv2.imread('etichetta1.jpg', cv2.IMREAD_COLOR)
# im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
hist1 = utils.hist_compute_orb(im)

im = cv2.imread('etichetta2.jpg', cv2.IMREAD_COLOR)
hist2 = utils.hist_compute_orb(im)

im = cv2.imread('etichetta3.jpg', cv2.IMREAD_COLOR)
hist3 = utils.hist_compute_orb(im)

while (True):
    ret, frame = cap.read()
    rects = []

    if ret:
        print("__________________________________")

        src = utils.hybrid_edge_detection_V2(frame)

        conts, heirarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        src_mask = np.zeros_like(src)
        hull_list = []

        for i in conts:
            rect = cv2.boundingRect(i)
            x, y, w, h = rect
            if w > 100 and h > 100:
                hull = cv2.convexHull(i)
                hull_list.append(hull)
                # creo contorni da dare alla funzione getline e un frame nero
                cv2.drawContours(src_mask, [hull], 0, (255, 255, 255), -1)

                rects.append(rect)

        #codice per fare output ROI varie
        #cv2.imshow('im', frame)
        #cv2.imshow('im', src_mask)
        #cv2.imshow('imh', houges)
        #cv2.imshow('cont_h', conts_h)

        outs = []
        masks = []

        # loop per estrarre e appendere a liste predifinite crop immagini
        for i in range(len(hull_list)):
            outs.append(utils.image_crop(frame, hull_list, i))
            masks.append(utils.image_crop_bin(src_mask, hull_list, i))

        # loop con calcolo histogramma, rectification e feature extraction con orb
        for idx in range(len(outs)):

            hist = utils.compute_histogram(outs[idx])
            entropy = utils.entropy(hist)
            print(entropy)
            utils.showImageAndStop("cropped", outs[idx])
            #cv2.imwrite("etichetta1.jpg",outs[idx])

            hist0 = utils.hist_compute_orb(outs[idx])
            distance1 = cv2.compareHist(hist0, hist1, cv2.HISTCMP_INTERSECT)
            distance2 = cv2.compareHist(hist0, hist2, cv2.HISTCMP_INTERSECT)
            distance3 = cv2.compareHist(hist0, hist3, cv2.HISTCMP_INTERSECT)

            print("distance1: {}".format(distance1))
            print("distance2: {}".format(distance2))
            print("distance3: {}".format(distance3))

            # con cv2.HISTCMP_BHATTACHARYYA circa 0.63 0.55 0.63
            #cv2.HISTCMP_INTERSECT
            #and distance2 >= 0.02
            if entropy >= 6 and distance1 <= 1.5 and distance2 <= 1.5 and distance3 <= 1.5:

                text, tmp = utils.detectKeyPoints(lista_immagini,lista_titoli, lista_stanze, outs[idx])
                if tmp != "":
                    room = tmp
                imm = masks[idx][0]
                out_bin_pad = cv2.copyMakeBorder(imm, 50, 50, 50, 50, 0)
                out_imm_pad = cv2.copyMakeBorder(outs[idx], 50, 50, 50, 50, 0)
                corners = utils.hougesLinesAndCorner(out_bin_pad)
                #utils.showImageAndStop("cropped",out_imm_pad)

                if len(corners) == 4:
                    p = utils.order_corners(corners)
                    #se order_corners non dà errore
                    if p!= 0:
                        warped = utils.rectify_image_2(out_imm_pad.shape[0], out_imm_pad.shape[1], out_imm_pad, p)
                        #se rectify_image_2 non daà errore
                        if not np.isscalar(warped):
                            utils.showImageAndStop('wrap', warped)
                            text, tmp = utils.detectKeyPoints(lista_immagini, lista_titoli, lista_stanze, warped)
                            if tmp != "":
                                room = tmp
                # corners = np.squeeze(corners, axis=1)
                # warped = utils.rectify_image(out_imm_pad,corners)
                utils.drawLabel(rects[idx][2], rects[idx][3], rects[idx][0], rects[idx][1], text, frame)

        #output dopo aver iterato su tutti gli out del frame
        print('esec yolo')
        frame = utils.fucking_yolo(frame, frame_height, frame_width)

        cv2.imshow("detect", frame)
        print(room)

        k = cv2.waitKey(5) & 0xFF
        if k == ord("q"):
            break

        # Write the frame into the file 'output.avi'
        out.write(frame)

    # Break the loop
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
cv2.waitKey(1)
