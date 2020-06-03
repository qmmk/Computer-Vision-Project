import cv2
import numpy as np
import utils
import yolo
import detect
import rectify
import os
import time
from matplotlib import pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import sys, getopt

no_gabor = True
rectify_image = False

# Get full command-line arguments
full_cmd_arguments = sys.argv

# Keep all but the first
argument_list = full_cmd_arguments[1:]
short_options = "ri:n"
long_options = ["rectify", "input", "no_gabor"]

try:
    arguments, values = getopt.getopt(argument_list,short_options ,long_options)
except getopt.error as err:
    print (str(err))
    sys.exit(2)

for current_argument, current_value in arguments:
    if current_argument in ("-n","--no_gabor"):
        print ("Enabling no gabor")
        no_gabor=False
    elif current_argument in ("-i", "--input"):
        if current_value is None:
            exit(2)
    elif current_argument in ("-r", "--rectify"):
        print("Enabling Rectify image")
        rectify_image = True


cap = cv2.VideoCapture(current_value)  #video_nome_lungo

if not cap.isOpened():
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))


out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

room = "Stanza generica"

template1 = cv2.imread('./match/match1.jpg', cv2.IMREAD_COLOR)
template2 = cv2.imread('./match/match2.jpg', cv2.IMREAD_COLOR)
template3 = cv2.imread('./match/match3.jpg', cv2.IMREAD_COLOR)
template4 = cv2.imread('./match/match4.png', cv2.IMREAD_COLOR)

dirname = 'rectifications'
if not os.path.exists(dirname):
    os.mkdir(dirname)

n_frame = 0

while (True):
    ret, frame = cap.read()
    if rectify_image:
        frame = utils.correct_distortion(frame,frame_height,frame_width)

    if ret:
        n_quadro = 0
        print(frame.shape)
        #frame = utils.correct_distortion(frame, frame_height, frame_width)
        #frame = utils.image_resize(frame, height=600)
        dict = []

        # DETECTION
        src = detect.hybrid_edge_detection_V2(frame,no_gabor)


        # CONTOURS
        rects, hulls, src_mask = detect.get_contours(src)


        # estrae gli indici delle roi senza intersezioni e rimuove gli indici di roi contenute in altre roi
        # utile per aumentare le performance e iterare solo su contorni certi
        # riduzioni falsi positivi
        listindexfree = utils.shrinkenCountoursList(hulls, frame, rects)

        blank = np.zeros_like(frame)
        for idk in listindexfree:
            cv2.drawContours(blank, hulls, idk, (255, 255, 255), 1)

        #utils.showImageAndStop("ROI",blank)

        # CROP
        outs, masks, green = detect.cropping_frame(frame, hulls, src_mask)

        # riduzione effettiva della lista di contorni e rect tramite index calcolati
        outs, rects = utils.reduceListOuts(outs, rects, listindexfree)


        sx = True
        # determianre orientamento
        for i in masks:
            corners = cv2.goodFeaturesToTrack(i, 4, 0.4, 80)
            print(corners)
            if len(corners) == 4 and i.shape > (150,150):
                sx,done = rectify.determineOrientation(i)
                if done:
                    break


        # FEATURE EXTRACTION
        for idx in range(len(outs)):
            hist = utils.compute_histogram(outs[idx])
            # entropy = utils.entropy(hist)
            # print(entropy)
            hist0 = utils.hist_compute_orb(green[idx])

            entropy = utils.entropy(hist0)
            print(entropy)

            # se è troppo piccolo scartalo
            if outs[idx].shape[0] >= template3.shape[0]:

                res1 = cv2.matchTemplate(outs[idx], template1, cv2.TM_CCORR_NORMED)
                res2 = cv2.matchTemplate(outs[idx], template2, cv2.TM_CCORR_NORMED)
                res3 = cv2.matchTemplate(outs[idx], template3, cv2.TM_CCORR_NORMED)
                res4 = cv2.matchTemplate(outs[idx], template4, cv2.TM_CCORR_NORMED)

                min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(res1)
                min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(res2)
                min_val3, max_val3, min_loc3, max_loc3 = cv2.minMaxLoc(res3)
                min_val4, max_val4, min_loc4, max_loc4 = cv2.minMaxLoc(res4)


                isBig = False
                if outs[idx].shape[0] > 300 and outs[idx].shape[1] > 300:
                    isBig = True

                if entropy >= 1.3 and ((max_val1 <= 0.96 and max_val2 <= 0.96 and max_val3 <= 0.96) or isBig):

                    imm = masks[idx]
                    out_bin_pad = cv2.copyMakeBorder(imm, 50, 50, 50, 50, 0)
                    out_imm_pad = cv2.copyMakeBorder(outs[idx], 50, 50, 50, 50, 0)

                    corners = rectify.hougesLinesAndCorner(out_bin_pad)

                    if len(corners) == 4:
                        local_orientation = rectify.determineOrientation(i)
                    else:
                        local_orientation = sx

                    print(local_orientation)

                    # RECTIFICATION
                    warped = 0
                    text, tmp, M, w, h = rectify.detectKeyPoints(outs[idx], local_orientation)
                    if tmp != "":
                        room = tmp
                    if not np.isscalar(M):
                        warped = cv2.warpPerspective(outs[idx], M, (w, h))
                        # utils.showImageAndStop("warped_sift",warped)  # `e qui che fa il display della imm warpata con sift

                    print("corner: {}".format(len(corners)))
                    print("text: {}".format(text))

                    if len(corners) == 4 and text == 'quadro':
                        p = rectify.order_corners(corners)
                        # se order_corners non dà errore
                        if p != 0:
                            ret = rectify.rectify_image_2(out_imm_pad.shape[0], out_imm_pad.shape[1], out_imm_pad, p)
                            #se rectify_image_2 non dà errore
                            if not np.isscalar(ret):
                                warped = ret
                                # utils.showImageAndStop("warped_corners", warped)
                                text, tmp, M, w, h = rectify.detectKeyPoints(warped,local_orientation)
                                if tmp != "":
                                    room = tmp
                                if not np.isscalar(M):
                                    warped = cv2.warpPerspective(warped, M, (w, h))
                                    # utils.showImageAndStop("warped_sift", warped)  # `e qui che fa il display della imm warpata con sift
                    if not np.isscalar(warped):
                        text_n = text.split('-')[0]
                        path = "./rectifications/" + str(n_frame) + "_" + str(n_quadro) + "_" + text_n + ".jpg"
                        cv2.imwrite(path, warped)
                        n_quadro += 1


                    dict.append({'texts': text, 'rects': rects[idx]})


        # PERSON

        dict = yolo.detect_person(frame, frame_height, frame_width, dict)
        frame = yolo.detect_eyes(frame)

        for di in dict:
            utils.drawLabel(di['rects'][2], di['rects'][3], di['rects'][0], di['rects'][1], di['texts'], frame)

        cv2.putText(frame, room, (20, frame_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        utils.showImageAndStop("detect", frame)
        # print(room)

        k = cv2.waitKey(5) & 0xFF
        if k == ord("q"):
            break

        # Write the frame into the file 'output.avi'
        out.write(frame)

        n_frame += 1

    # Break the loop
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
cv2.waitKey(1)
