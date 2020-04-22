import cv2
import numpy as np

video_stronzo = 'VIRB0391.MP4'
# 'VIRB0407.MP4'
# 'GOPR2051.MP4'
cap = cv2.VideoCapture('VIRB0407.MP4')

if (cap.isOpened() == False):
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

mask_2 = np.zeros((frame_height + 2, frame_width + 2), np.uint8)

out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

def drawLabel(w, h, x, y, text, frame):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (120, 0, 0), 2)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

def affine_erode_dilate(input,kernel):
    erosion = cv2.erode(input, kernel, iterations=1)
    dilate = cv2.dilate(erosion, kernel, iterations=2)
    return dilate

#kernel usati
kernel_3 = np.ones((3, 3), np.uint8)
kernel = np.ones((3, 3), np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)
g_kernel = cv2.getGaborKernel((25, 25), 6.5, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
#term_crit = ( cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1 )


#global used variable
color = (255, 255, 255)

while (True):
    ret, frame = cap.read()
    rects = {}

    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        erode =cv2.erode(thresh,kernel_5, iterations=1)

        conts, heirarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        conts_shirnk = []
        hull_list = []

        for i in conts:
            rect = cv2.boundingRect(i)
            x, y, w, h = rect
            if w > 150 and h > 150:
                conts_shirnk.append(i)
                hull = cv2.convexHull(i)
                hull_list.append(hull)
                cv2.drawContours(frame, [hull], 0, (0, 255, 0), -1)


        cv2.drawContours(frame, hull_list, -1, (0, 0, 255))

        cv2.imshow('Contours', erode)

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