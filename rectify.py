import cv2
import numpy as np
import scipy.spatial.distance
import math
import utils
from PIL import Image


lista_titoli, lista_immagini, lista_stanze = utils.carica_lista_cvs()


def order_corners(corners):
    p = []
    # p.append((corners[2][0][0], corners[2][0][1]))
    # p.append((corners[3][0][0], corners[3][0][1]))
    # p.append((corners[1][0][0], corners[1][0][1]))
    # p.append((corners[0][0][0], corners[0][0][1]))
    sumx = corners[2][0][0] + corners[3][0][0] + corners[1][0][0] + corners[0][0][0]
    sumy = corners[2][0][1] + corners[3][0][1] + corners[1][0][1] + corners[0][0][1]
    medx = sumx / 4
    medy = sumy / 4

    for c in corners[:]:
        # bottom left
        if c[0][0] < medx and c[0][1] < medy:
            bl = (c[0][0], c[0][1])
        # bottom right
        if c[0][0] > medx and c[0][1] < medy:
            br = (c[0][0], c[0][1])
        # top left
        if c[0][0] < medx and c[0][1] > medy:
            tl = (c[0][0], c[0][1])
        # top right
        if c[0][0] > medx and c[0][1] > medy:
            tr = (c[0][0], c[0][1])
    try:
        p.append(bl)
        p.append(br)
        p.append(tl)
        p.append(tr)
    except UnboundLocalError:
        print("The corner are wrong")
        return 0

    return p


def rectify_image_2(rows, cols, img, p):
    # image center
    u0 = (cols) / 2.0
    v0 = (rows) / 2.0
    # widths and heights of the projected image
    w1 = scipy.spatial.distance.euclidean(p[0], p[1])
    w2 = scipy.spatial.distance.euclidean(p[2], p[3])

    h1 = scipy.spatial.distance.euclidean(p[0], p[2])
    h2 = scipy.spatial.distance.euclidean(p[1], p[3])

    w = max(w1, w2)
    h = max(h1, h2)

    # visible aspect ratio
    ar_vis = float(w) / float(h)

    # make numpy arrays and append 1 for linear algebra
    m1 = np.array((p[0][0], p[0][1], 1)).astype('float32')
    m2 = np.array((p[1][0], p[1][1], 1)).astype('float32')
    m3 = np.array((p[2][0], p[2][1], 1)).astype('float32')
    m4 = np.array((p[3][0], p[3][1], 1)).astype('float32')

    # calculate the focal disrance
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)

    n2 = k2 * m2 - m1
    n3 = k3 * m3 - m1

    n21 = n2[0]
    n22 = n2[1]
    n23 = n2[2]

    n31 = n3[0]
    n32 = n3[1]
    n33 = n3[2]

    # per evitare divisioni per 0
    try:
        f = math.sqrt(np.abs((1.0 / (n23 * n33)) * ((n21 * n31 - (n21 * n33 + n23 * n31) * u0 + n23 * n33 * u0 * u0) + (
                n22 * n32 - (n22 * n33 + n23 * n32) * v0 + n23 * n33 * v0 * v0))))

        A = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]]).astype('float32')

        At = np.transpose(A)
        Ati = np.linalg.inv(At)
        Ai = np.linalg.inv(A)

        # calculate the real aspect ratio
        ar_real = math.sqrt(np.dot(np.dot(np.dot(n2, Ati), Ai), n2) / np.dot(np.dot(np.dot(n3, Ati), Ai), n3))

        if ar_real < ar_vis:
            W = int(w)
            H = int(W / ar_real)
        else:
            H = int(h)
            W = int(ar_real * H)

        pts1 = np.array(p).astype('float32')
        pts2 = np.float32([[0, 0], [W, 0], [0, H], [W, H]])

        # project the image with the new w/h
        M = cv2.getPerspectiveTransform(pts1, pts2)

        dst = cv2.warpPerspective(img, M, (W, H))

        # cv2.imshow('img', img)
        # cv2.imshow('dst', dst)

        # cv2.waitKey(0)
    except ValueError:
        print("error in the formula")
        return 0

    return dst


def rectify_image_with_correspondences(im, p1, p2, w, h):
    m, status = cv2.findHomography(p1, p2)
    warped = cv2.warpPerspective(im, m, (w, h))

    return warped


def getLine(edges, frame):
    # get contours

    minLineLength = 100

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)
    if lines is None:
        return

    N = lines.shape[0]
    for i in range(N):
        x1 = lines[i][0][0]
        y1 = lines[i][0][1]
        x2 = lines[i][0][2]
        y2 = lines[i][0][3]

        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    '''
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    '''


def ORB(im1, im2, titolo_immagine):
    # Initiate SIFT detector
    orb = cv2.ORB_create()
    # cv2.imshow("im1", im1)
    # cv2.imshow("im2", im2)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(im1, None)
    kp2, des2 = orb.detectAndCompute(im2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    if (des1 is None) or (des2 is None):
        return False, 0, 0, 0, 100000

    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    good = []
    retkp1 = []
    retkp2 = []
    ngood = 10


    for m in matches:
        if m.distance < 40:  # 50
            good.append(m)
            # Get the matching keypoints for each of the images
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx
            (x1, y1) = kp1[img1_idx].pt
            (x2, y2) = kp2[img2_idx].pt
            retkp1.append((x1, y1))
            retkp2.append((x2, y2))

    if len(good) >= ngood:
        good = sorted(good, key=lambda x: x.distance)
        score = sum(x.distance for x in good[:ngood])
        # print("{} -> score: {}".format(titolo_immagine, score))

        if score < 350:  # 230
            """
            hist1 = utils.hist_compute_orb(im1)
            hist2 = utils.hist_compute_orb(im2)
            distance1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)
            print('distance' + str(distance1))
            if distance1 <= 1.5:
                print('check') 
            """
            img3 = cv2.drawMatches(im1, kp1, im2, kp2, good[:ngood], None, flags=2)
            utils.showImageAndStop(titolo_immagine,img3)
            return True, good, retkp1, retkp2, score
        else:
            return False, 0, 0, 0, 100000
    else:
        return False, 0, 0, 0, 100000


def detectKeyPoints(img_rgb):
    min_idx = -1
    min_score = 100000
    text = "quadro"
    room = ""
    for it in range(len(lista_immagini) - 1):
        # Read the main image
        titolo_quadro = lista_titoli[it + 1]
        immage_template = "./template/" + lista_immagini[it + 1]
        stanza = lista_stanze[it + 1]
        template = cv2.imread(immage_template, 1)

        is_detected, matches, ret_kp1, ret_kp2, score = ORB(img_rgb, template, titolo_quadro)
        if score < min_score:
            min_score = score
            text = "{} - score: {}".format(titolo_quadro, score)
            room = "Stanza n.{}".format(stanza)

            min_idx = it
            array1 = np.array((ret_kp1), dtype=np.float32)
            array2 = np.array((ret_kp2), dtype=np.float32)

    if min_score < 100000:
        id = min_idx
        print("idx" + str(id))
        warped = rectify_image_with_correspondences(img_rgb, array2[:8], array1[:8], 1000, 1000)
        #utils.showImageAndStop(text, warped)

    return text, room


def hougesLinesAndCorner(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180.0, 100, np.array([]), 0, 0)
    out_line = np.zeros_like(image)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(out_line, (x1, y1), (x2, y2), (255, 255, 255), 1)

    # img_gray = cv2.cvtColor(out_line, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('out', out_line)
    # cv2.waitKey()

    corners = cv2.goodFeaturesToTrack(out_line, 4, 0.4, 80)

    if corners is not None:
        corners = np.int0(corners)

        for i in corners:
            x, y = i.ravel()
            cv2.circle(out_line, (x, y), 3, 255, -1)

    else:
        corners = []
        return corners

    # showImageAndStop('hough', out_line)

    return corners