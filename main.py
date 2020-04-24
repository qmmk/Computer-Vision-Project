import cv2
import numpy as np
import utils

video_stronzo = './videos/VIRB0391.MP4'
video_normale = './videos/VIRB0414.MP4'
video_tondo = './videos/GOPR2051.MP4'
video_comune = './videos/VIRB0407.MP4'
cap = cv2.VideoCapture(video_comune)

if (cap.isOpened() == False):
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

# global used variable
color = (255, 255, 255)

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

                ''' 
                for j in vertex:
                    cv2.circle(frame, tuple(j[0]),3,(255,0,0),-1)
                '''
        #cv2.imshow('im', canny)
        cv2.imshow('imh', houges)

        outs = []
        masks = []

        for i in range(len(hull_list)):
            outs = utils.image_crop(frame, hull_list, i)


            for idx in range(len(outs)):

                hist = utils.compute_histogram(outs[idx])
                entropy = utils.entropy(hist)
                pts = np.squeeze(vertex[i], axis=1)
                print(entropy)
                if entropy >= 1:
                    #cv2.imshow(str(i), outs[idx])
                    # rectification
                    warped = utils.rectify_image(outs[idx], pts)
                    cv2.imshow(str(i), warped)
                    


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
