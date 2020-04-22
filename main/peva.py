import cv2
import numpy as np

# Create a VideoCapture object
# apre il video da

video_stronzo = 'VIRB0391.MP4'
video_normale = 'VIRB0407.MP4'
video_tondo = 'GOPR2051.MP4'
cap = cv2.VideoCapture(video_tondo)

#controlla se `e stato aperto correttamente
# Check if camera opened successfully
if (cap.isOpened() == False):
    print("Unable to read camera feed")

# Default resolutions of the frame are obtained.The default resolutions are system dependent.
# We convert the resolutions from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

mask_2 = np.zeros((frame_height+2, frame_width+2), np.uint8)

# definizone output
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

def contains_vertical(r1, r2):
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return x1 <= x2 < x1 + w1 and x1 <= x2 + w2 < x1 + w1


class ColourBounds:
    def __init__(self, rgb):
        hsv = cv2.cvtColor(np.uint8([[[rgb[2], rgb[1], rgb[0]]]]), cv2.COLOR_BGR2HSV).flatten()

        lower = [hsv[0] - 10]
        upper = [hsv[0] + 10]

        if lower[0] < 0:
            lower.append(179 + lower[0]) # + negative = - abs
            upper.append(179)
            lower[0] = 0
        elif upper[0] > 179:
            lower.append(0)
            upper.append(upper[0] - 179)
            upper[0] = 179

        self.lower = [np.array([h, 100, 100]) for h in lower]
        self.upper = [np.array([h, 255, 255]) for h in upper]

colourMap = {
        "quadro": ColourBounds((150, 130, 100))
}

def drawLabel(w, h, x, y, text, frame):
    cv2.rectangle(frame,(x,y),(x+w,y+h),(120,0,0),2)
    cv2.putText(frame, text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)


kernel = np.ones((3,3),np.uint8)
kernel2 = np.ones((5,5),np.uint8)

while (True):
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rects = {}


    if ret == True:

        for name, colour in colourMap.items():

            mask = cv2.inRange(hsv, colour.lower[0], colour.upper[0])

            if len(colour.lower) == 2:
                mask = mask | cv2.inRange(hsv, colour.lower[1], colour.upper[1])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #g_kernel = cv2.getGaborKernel((15, 15), 6.5, np.pi / 4, 10.0, 0.5, 0, ktype=cv2.CV_32F) #se usi questo kernel per entrambi è più stabile ma non prende quadro sbiadito

            g_kernel = cv2.getGaborKernel((15, 15), 8.0, np.pi / 4, 10.0, 0.5, 0.5, ktype=cv2.CV_32F)
            g_kernel2 = cv2.getGaborKernel((15, 15), 8.5, np.pi / 4, 10, 0.5, 0, ktype=cv2.CV_32F)
            gray = cv2.filter2D(gray, cv2.CV_8UC3, g_kernel)
            gray = cv2.GaussianBlur(gray, (7, 7), 15)
            gray = cv2.GaussianBlur(gray, (7, 7), 15)
            gray = cv2.GaussianBlur(gray, (7, 7), 15)

            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
            edges = cv2.bitwise_not(edges)
            erosion = cv2.erode(edges, kernel, iterations=2)
            erosion = cv2.medianBlur(erosion, 3)
            erosion_f = cv2.filter2D(erosion, cv2.CV_8UC3, g_kernel2)
            dilatation_out = cv2.dilate(erosion_f, kernel2, iterations=7)
            erosion2 = cv2.erode(dilatation_out, kernel2, iterations=2)
            src_dilat = erosion2.copy()

            conts, heirarchy = cv2.findContours(src_dilat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


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
                    rects['quadro'] = rect
                    drawLabel(w, h, x, y, name, frame)

            #cv2.drawContours(frame, conts_shirnk, -1, (0, 255, 0), 3)
            cv2.drawContours(frame, hull_list,-1, (0, 0, 255))

            cv2.imshow('Contours', frame)

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

# Closes all the framessrc_dilat
cv2.destroyAllWindows()
cv2.waitKey(1)