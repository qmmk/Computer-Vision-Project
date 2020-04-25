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
kernel = np.ones((3,3),np.uint8)
kernel2 = np.ones((5,5),np.uint8)
kernel3 = np.ones((7,7),np.uint8)


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
                    print(j[0].shape)
                    cv2.circle(houges, tuple(np.squeeze(j[0], axis=0)),10,(100, 0, 0),-1)
                '''

        #cv2.imshow('im', src_mask)
        #cv2.imshow('canny', canny)
        #cv2.imshow('imh', houges)
        #cv2.imshow('cont_h', conts_h)

        outs = []
        masks = []

        for i in range(len(hull_list)):
            outs = utils.image_crop(frame, hull_list, i)
            outs_bin = utils.image_crop_bin(src_mask, hull_list, i)

            for idx in range(len(outs)):

                hist = utils.compute_histogram(outs[idx])
                entropy = utils.entropy(hist)
                #pts = np.squeeze(vertex[i], axis=1)
                if entropy >= 1:
                    outs_bin[idx] = utils.add_margin(outs_bin[idx], 10,10, bin=True)
                    outs[idx] = utils.add_margin(outs[idx], 10, 10)

                    dst = cv2.cornerHarris(outs_bin[idx], 2, 3, 0.02)
                    dst = cv2.dilate(dst, None)
                    ret, dst = cv2.threshold(dst, 0.1 * dst.max(), 255, 0)
                    dst = np.uint8(dst)
                    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                    corners = cv2.cornerSubPix(houges, np.float32(centroids), (5, 5), (-1, -1), criteria)
                    corners = np.around(corners)
                    outs[idx][dst > 0.01 * dst.max()] = [0, 0, 255]

                    cv2.imshow(str(i), outs[idx])
                    # rectification
                    warped = utils.rectify_image(outs[idx], corners)
                    cv2.imshow(str(i)+"_warp", warped)
                    


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
