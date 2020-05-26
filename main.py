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
video_fish_eye = './videos/GOPR5818.MP4'  # da scaricare
video_statua_fish_eye = './videos/GOPR5831.MP4'  # da scaricare
video_statua_negra = "./videos/IMG_7854.MOV"
video_statue_col = "./videos/IMG_9630.MOV"
video_statue_white = "./videos/IMG_4080.MOV"
video_statue_white_t = "./videos/IMG_4080_Trim.mp4"
video_k = "./videos/Nuovi/GOPR5827.MP4"
video_o = "./videos/Nuovi/GOPR5825.MP4"
video_q = "./videos/Nuovi/GOPR2049.MP4"
video_pi = "./videos/Nuovi/IMG_9621.MOV"
video_a = "./videos/Nuovi/IMG_9628.MOV"
video_b = "./videos/Nuovi/IMG_4076.MOV"
video_d = "./videos/Nuovi/IMG_4074.MOV"
uomo_cappello = "./videos/Nuovi/IMG_2653.MOV"
vid = "./videos/Nuovi/GOPR5828.MP4"
video_zoom = "./videos/Nuovi/VID_20180529_112627.mp4"
video_largo = "./videos/Nuovi/20180206_113800.mp4"
video_cornice = "./videos/Nuovi/20180206_111931.mp4"
video_no_cornice = "./videos/Nuovi/IMG_7852.MOV"
video_tipo = "./videos/Nuovi/IMG_9622.MOV"
video_ll = "./videos/Nuovi/VID_20180529_112706.mp4"
video_corridoio = "./videos/Nuovi/VIRB0394.MP4"
video_distorto = "./videos/GOPR5830.MP4"
video_rombi = "./videos/GOPR5825.MP4"
video_prete = "./videos/Nuovi/GOPR5826.MP4"
video_gesu = "./videos/Nuovi/20180206_113600_Trim.mp4"
video_asd = "./videos/VIRB0416.MP4"
video_uomo_trimmato = "./videos/Nuovi/uomo_trimmato.mp4"
video_stanza_ampia = "./videos/20180206_114720_Trim.mp4"

cap = cv2.VideoCapture(video_trim)

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


while (True):
    ret, frame = cap.read()

    if ret:
        #frame = utils.correct_distortion(frame, frame_height, frame_width)

        dict = []

        # DETECTION
        src = detect.hybrid_edge_detection_V2(frame)


        # CONTOURS
        rects, hulls, src_mask = detect.get_contours(src)


        # estrae gli indici delle roi senza intersezioni e rimuove gli indici di roi contenute in altre roi
        # utile per aumentare le performance e iterare solo su contorni certi
        # riduzioni falsi positivi
        listindexfree = utils.shrinkenCountoursList(hulls, frame, rects)

        blank = np.zeros_like(frame)
        for idk in listindexfree:
            cv2.drawContours(blank, hulls, idk, (255, 255, 255), 1)

        utils.showImageAndStop("ROI",blank)

        # CROP
        outs, masks, green = detect.cropping_frame(frame, hulls, src_mask)

        # riduzione effettiva della lista di contorni e rect tramite index calcolati
        outs, rects = utils.reduceListOuts(outs, rects, listindexfree)


        sx = True
        # determianre orientamento
        for i in masks:
            corners = cv2.goodFeaturesToTrack(i, 4, 0.4, 80)
            if len(corners) == 4 and i.shape > (150,150):
                sx = rectify.determineOrientation(i)
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

                #utils.showImageAndStop("cropped", outs[idx])
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

                    # RECTIFICATION
                    text, tmp = rectify.detectKeyPoints(outs[idx],sx)
                    if tmp != "":
                        room = tmp
                    imm = masks[idx]
                    out_bin_pad = cv2.copyMakeBorder(imm, 50, 50, 50, 50, 0)
                    out_imm_pad = cv2.copyMakeBorder(outs[idx], 50, 50, 50, 50, 0)

                    corners = rectify.hougesLinesAndCorner(out_bin_pad)

                    print("corner: {}".format(len(corners)))
                    print("text: {}".format(text))

                    if len(corners) == 4 and text == 'quadro':
                        p = rectify.order_corners(corners)
                        # se order_corners non dà errore
                        if p != 0:
                            warped = rectify.rectify_image_2(out_imm_pad.shape[0], out_imm_pad.shape[1], out_imm_pad, p)
                            #se rectify_image_2 non dà errore
                            if not np.isscalar(warped):
                                utils.showImageAndStop("warped_corners", warped)
                                text, tmp = rectify.detectKeyPoints(warped,sx)
                                if tmp != "":
                                    room = tmp
                    dict.append({'texts': text, 'rects': rects[idx]})


        # PERSON

        detected, rects_yolo, label_yolo = yolo.detect_person(frame, frame_height, frame_width)
        frame, rects_dec_eyes = yolo.detect_eyes(frame, detected)


        # codice per controllo/aggiunta/rimozione rettangoli oggetti detctetati
        for r in range(0, len(rects_yolo)):
            dict.append({'texts': label_yolo[r], 'rects': rects_yolo[r]})
        for r in range(0, len(rects_dec_eyes)):
            dict.append({'texts': 'statua', 'rects': rects_dec_eyes[r]})

        kkk = dict[:]
        kk = []
        for a in kkk:
            kk.append(a['rects'])

        listindexyolo = list(range(0, len(dict)))
        listindexnoinside_yolo = utils.checkInside(kk, listindexyolo)
        listindexnoinside_yolo.sort()

        for index in reversed(listindexyolo):
            if index in listindexnoinside_yolo:
                a=dict.pop(index)
        # fine controllo/aggiunta/rimozione

        for di in dict:
            utils.drawLabel(di['rects'][2], di['rects'][3], di['rects'][0], di['rects'][1], di['texts'], frame)

        utils.showImageAndStop("detect", frame)
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
