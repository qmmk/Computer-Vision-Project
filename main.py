import cv2
import numpy as np
import utils

video_stronzo = './videos/VIRB0391.MP4'
video_normale = './videos/VIRB0407.MP4'
video_tondo = './videos/GOPR2051.MP4'
cap = cv2.VideoCapture(video_stronzo)

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

        src = utils.otsu(frame)

        # ADAPTIVE
        # src_dilat = utils.adaptive(frame)
        # for src in src_dilat:         # loop on color

        conts, heirarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        hull_list = []

        for i in conts:
            rect = cv2.boundingRect(i)
            x, y, w, h = rect
            if w > 150 and h > 150:
                hull = cv2.convexHull(i)
                hull_list.append(hull)
                vertex = cv2.approxPolyDP(i, 0.009 * cv2.arcLength(i, True), True)
                ''' 
                for j in vertex:
                    cv2.circle(frame, tuple(j[0]),3,(255,0,0),-1)
                '''
        # cv2.drawContours(frame, hull_list, -1, (0, 0, 255))

        outs = []
        masks = []
        for i in range(len(hull_list)):
            outs = utils.image_crop(frame, hull_list, i)

            for idx in range(len(outs)):
                hist = utils.compute_histogram(outs[idx])
                entropy = utils.entropy(hist)
                pts = np.squeeze(vertex, axis=1)
                if entropy >= 5:
                    # rectification 
                    warped = utils.rectify_image(outs[i], pts)

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
