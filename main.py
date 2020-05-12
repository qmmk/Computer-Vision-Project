import cv2
import numpy as np
import utils
import yolo
import detect
import rectify
import time
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, exposure

video_stronzo = './videos/VIRB0391.MP4'
video_normale = './videos/VIRB0414.MP4'
video_tondo = './videos/GOPR2051.MP4'
video_comune = './videos/VIRB0407.MP4'
video_madonna_bimbo = './videos/VIRB0392.MP4'
video_boh = "./videos/VID_20180529_112440.mp4"  # ci vuole gabor filter da 30
video_comune_trim = "./videos/trim.mp4"
video_trim = "./videos/trimsss.mp4"
video_diverso = './videos/VIRB0415.MP4'
video_gente = './videos/GOPR1940.MP4'
video_1 = './videos/GOPR1928.MP4'
video_2 = './videos/GOPR1947.MP4'
video_3 = './videos/GOPR2039.MP4'
video_nome_lungo = './videos/VID_20180529_112539.mp4'
video_persone_s = './videos/trim_2.mp4'
video_facce = "./videos/20180206_114604.mp4"
video_sono_le_11 = './videos/IMG_4082.MOV'

cap = cv2.VideoCapture(video_persone_s)

if not cap.isOpened():
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

room = "Stanza generica"

im = cv2.imread('etichetta1.jpg', cv2.IMREAD_COLOR)
hist1 = utils.hist_compute_orb(im)

im = cv2.imread('etichetta2.jpg', cv2.IMREAD_COLOR)
hist2 = utils.hist_compute_orb(im)

im = cv2.imread('etichetta3.jpg', cv2.IMREAD_COLOR)
hist3 = utils.hist_compute_orb(im)

while (True):
    ret, frame = cap.read()

    if ret:

        # DETECTION
        src = detect.hybrid_edge_detection_V2(frame)

        # CONTOURS
        rects, hulls, src_mask = detect.get_contours(src)

        #estrae gli indici delle roi senza intersezioni e rimuove gli indici di roi contenute in altre roi
        #utile per aumentare le performance e iterare solo su contorni certi
        #riduzioni falsi positivi
        listindexfree = utils.shrinkenCountoursList(hulls,frame,rects)


        blank = np.zeros_like(frame)
        for idk in listindexfree:
            cv2.drawContours(blank, hulls, idk, (255, 255, 255), -1)

        utils.showImageAndStop('ROI',blank)

        # CROP
        outs, masks, green = detect.cropping_frame(frame, hulls, src_mask)

        # riduzione effettiva della lista di contorni e rect tramite index calcolati
        outs,rects = utils.reduceListOuts(outs,rects,listindexfree)

        # FEATURE EXTRACTION
        for idx in range(len(outs)):

            hist0 = utils.hist_compute_orb(green[idx])
            entropy = utils.entropy(hist0)
            print('entropia: ' + str(entropy))
            distance1 = cv2.compareHist(hist0, hist1, cv2.HISTCMP_BHATTACHARYYA)
            distance2 = cv2.compareHist(hist0, hist2, cv2.HISTCMP_BHATTACHARYYA)
            distance3 = cv2.compareHist(hist0, hist3, cv2.HISTCMP_BHATTACHARYYA)

            # con cv2.HISTCMP_BHATTACHARYYA circa 0.63 0.55 0.63
            # cv2.HISTCMP_INTERSECT
            # and distance2 >= 0.02
            print('d1: ' + str(distance1))
            print('d2: ' + str(distance2))
            print('d3: ' + str(distance3))

            #and (distance1 > 0.6 and distance2 > 0.7 and distance3 > 0.6)

            if entropy >= 2.5:

                # RECTIFICATION
                text, tmp = rectify.detectKeyPoints(outs[idx])
                if tmp != "":
                    room = tmp
                imm = masks[idx]
                out_bin_pad = cv2.copyMakeBorder(imm, 50, 50, 50, 50, 0)
                out_imm_pad = cv2.copyMakeBorder(outs[idx], 50, 50, 50, 50, 0)
                corners = rectify.hougesLinesAndCorner(out_bin_pad)
                utils.showImageAndStop("cropped",out_imm_pad)


                if len(corners) == 4 and text == 'quadro':
                    p = rectify.order_corners(corners)
                    # se order_corners non dà errore
                    if p != 0:
                        warped = rectify.rectify_image_2(out_imm_pad.shape[0], out_imm_pad.shape[1], out_imm_pad, p)
                        # se rectify_image_2 non dà errore
                        if not np.isscalar(warped):
                            text, tmp = rectify.detectKeyPoints(warped)
                            if tmp != "":
                                room = tmp

                utils.drawLabel(rects[idx][2], rects[idx][3], rects[idx][0], rects[idx][1], text, frame)




        # PERSON
        frame = yolo.detect_person(frame, frame_height, frame_width)
        frame = yolo.detect_eyes(frame)
        cv2.imshow("detect", frame)
        # print(room)

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
