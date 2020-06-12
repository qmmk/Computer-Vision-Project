import cv2
import numpy as np
import utils
import yolo
import detect
import rectify
import os
import sys, getopt
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import prediction

SVM = prediction.setup()

no_gabor = True
rectify_image = False

# Get full command-line arguments
full_cmd_arguments = sys.argv

# Keep all but the first
argument_list = full_cmd_arguments[1:]
short_options = "ri:n"
long_options = ["rectify", "input", "no_gabor"]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    print(str(err))
    sys.exit(2)

for current_argument, current_value in arguments:
    if current_argument in ("-n", "--no_gabor"):
        print("Enabling no gabor")
        no_gabor = False
    elif current_argument in ("-i", "--input"):
        if current_value is None:
            exit(2)
    elif current_argument in ("-r", "--rectify"):
        print("Enabling Rectify image")
        rectify_image = True

# INITIALIZE RESNET
feature_vectors = utils.carica_feature_csv()
resnet18 = models.resnet18(pretrained=True)
layer = resnet18._modules.get('avgpool')
resnet18.eval()
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

video = './videos/VIRB0415.MP4'
cap = cv2.VideoCapture(video)

if not cap.isOpened():
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

room = "Stanza generica"
dirname = 'rectifications'

if not os.path.exists(dirname):
    os.mkdir(dirname)

n_frame = 0

while (True):
    ret, frame = cap.read()

    if frame.shape[0] > frame.shape[1]:
        if frame.shape[0] > 1080:
            frame = utils.image_resize(frame, height=1080)
    elif frame.shape[1] > frame.shape[0]:
        if frame.shape[1] > 1920:
            frame = utils.image_resize(frame, width=1920)

    if rectify_image:
        frame = utils.correct_distortion(frame, frame_height, frame_width)

    if ret:
        n_quadro = 0
        dict = []
        res = []

        # DETECTION
        src = detect.hybrid_edge_detection_V2(frame, no_gabor)

        # CONTOURS
        rects, hulls, src_mask = detect.get_contours(src)

        # indici roi senza intersezioni e no contenute
        listindexfree = utils.shrinkenCountoursList(hulls, frame, rects)

        # CROP
        outs, masks, green = detect.cropping_frame(frame, hulls, src_mask)

        outs, rects = utils.reduceListOuts(outs, rects, listindexfree)

        # orientamento sx/dx
        sx = True
        for i in masks:
            corners = cv2.goodFeaturesToTrack(i, 4, 0.4, 80)
            if corners is not None and len(corners) == 4 and i.shape > (150, 150):
                sx, done = rectify.determineOrientation(i)
                if done:
                    break

        # FEATURE EXTRACTION
        for idx in range(len(outs)):
            hist = utils.hist_compute_orb(green[idx])
            entropy = utils.entropy(hist)

            # COSINE SIMILARITY
            im_pil = Image.fromarray(outs[idx])
            vec = detect.get_feature_vector(im_pil, scaler, to_tensor, normalize, layer, resnet18)

            prediction_svm = prediction.check(SVM, vec)

            if entropy >= 1.3 and prediction_svm:

                out_bin_pad = cv2.copyMakeBorder(masks[idx], 50, 50, 50, 50, 0)
                out_imm_pad = cv2.copyMakeBorder(outs[idx], 50, 50, 50, 50, 0)

                corners = rectify.hougesLinesAndCorner(out_bin_pad)

                if len(corners) == 4:
                    local_orientation = rectify.determineOrientation(i)
                else:
                    local_orientation = sx

                # RECTIFICATION
                warped = 0
                text, room, M, w, h = rectify.detectKeyPoints(outs[idx], local_orientation)
                if not np.isscalar(M):
                    warped = cv2.warpPerspective(outs[idx], M, (w, h))

                if len(corners) == 4 and text == 'quadro':
                    p = rectify.order_corners(corners)
                    if p != 0:
                        ret = rectify.rectify_image(out_imm_pad.shape[0], out_imm_pad.shape[1], out_imm_pad, p)
                        if not np.isscalar(ret):
                            warped = ret
                            text, room, M, w, h = rectify.detectKeyPoints(warped, local_orientation)
                            if not np.isscalar(M):
                                warped = cv2.warpPerspective(warped, M, (w, h))

                if not np.isscalar(warped):
                    res.append({'not': outs[idx], 'yes': warped})

                dict.append({'texts': text, 'rects': rects[idx]})

        # PERSON
        dict = yolo.detect_person(frame, frame_height, frame_width, dict)
        frame = yolo.detect_eyes(frame)

        for di in dict:
            utils.drawLabel(di['rects'][2], di['rects'][3], di['rects'][0], di['rects'][1], di['texts'], frame)

        utils.display(room, res, frame, src_mask)

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
