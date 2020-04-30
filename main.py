import cv2
import numpy as np
import utils
import time

video_stronzo = './videos/VIRB0391.MP4'
video_normale = './videos/VIRB0414.MP4'
video_tondo = './videos/GOPR2051.MP4'
video_comune = './videos/VIRB0407.MP4'
video_madonna_bimbo = './videos/VIRB0392.MP4'

cap = cv2.VideoCapture(video_madonna_bimbo)

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

lista_titoli, lista_immagini = utils.carica_lista_cvs()


while (True):
    ret, frame = cap.read()
    rects = {}

    if ret:

        src = utils.hybrid_edge_detection(frame)

        # ADAPTIVE
        # src_dilat = utils.adaptive(frame)
        # for src in src_dilat:         # loop on color

        conts, heirarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        src_mask = np.zeros_like(src)
        houges = np.zeros_like(src)
        hull_list = []
        vertex = []

        for i in conts:
            rect = cv2.boundingRect(i)
            x, y, w, h = rect
            if w > 100 and h > 100:
                hull = cv2.convexHull(i)
                hull_list.append(hull)
                vertex.append(cv2.approxPolyDP(i, 0.009 * cv2.arcLength(i, True), True))

                # creo contorni da dare alla funzione getline e un frame nero
                cv2.drawContours(src_mask, [hull], 0, (255, 255, 255), -1)
                canny = cv2.Canny(src_mask, 20, 200,apertureSize = 3)
                utils.getLine(canny,houges)


        # codice per fare output ROI varie
        #cv2.imshow('im', src_mask)
        #cv2.imshow('canny', canny)
        #cv2.imshow('imh', houges)
        #cv2.imshow('cont_h', conts_h)

        outs = []
        masks = []
        cannys = []

        # loop per estrarre e appendere a liste predifinite crop immagini
        for i in range(len(hull_list)):
            outs.append(utils.image_crop(frame, hull_list, i))
            masks.append(utils.image_crop_bin(src_mask, hull_list, i))

        # loop con calcolo histogramma, rectification e feature extraction con orb
        for idx in range(len(outs)):
            hist = utils.compute_histogram(outs[idx])
            entropy = utils.entropy(hist)

            if entropy >= 3:
                imm = masks[idx][0]
                out_bin_pad = cv2.copyMakeBorder(imm,20,20,20,20,0)
                out_imm_pad = cv2.copyMakeBorder(outs[idx],20,20,20,20,0)

                corners = cv2.goodFeaturesToTrack(out_bin_pad,4,0.4,80)
                corners = np.int0(corners)
                print(np.squeeze(corners, axis=1).shape)

                if corners is not None:
                    for i in corners:
                        x,y = i.ravel()
                        cv2.circle(out_imm_pad,(x,y),3,255,-1)

                if len(corners) == 4:
                    corners = np.squeeze(corners, axis=1)
                    warped = utils.rectify_image(out_imm_pad, corners)
                    warped = cv2.copyMakeBorder(warped, 50, 50, 50, 50, 0)
                    #cv2.imshow(str(idx) + "_warp", out_imm_pad)


                    for it in range(len(lista_immagini)-1):
                        # Read the main image
                        titolo_quadro = lista_titoli[it+1]
                        immage_template = "./template/"+lista_immagini[it+1]
                        img_rgb = warped
                        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
                        template = cv2.imread(immage_template,0)

                        is_detected, matches = utils.ORB(img_gray,template,titolo_quadro)
                        if is_detected:
                            print(titolo_quadro)

        k = cv2.waitKey(5) & 0xFF
        if k == ord("q"):
            break

        # Write the frame into the file 'output.avi'
        # out.write(frame)

    # Break the loop
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
cv2.waitKey(1)
