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
        if current_value is not None:
            video = current_value
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


cap = cv2.VideoCapture(video)
frame_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if not cap.isOpened():
    print("Unable to read camera feed")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1920, 1080))

dirname = 'rectifications'

if not os.path.exists(dirname):
    os.mkdir(dirname)

n_frame = 0
n_quadro = 0
tmp = "0"
res = []

while (True):
    ret, frame = cap.read()

    if ret:

        if rectify_image:
            frame = utils.correct_distortion(frame, frame_height, frame_width)

        dict = []
        check = True

        roi = np.zeros_like(frame)

        # DETECTION
        src = detect.hybrid_edge_detection(frame, no_gabor)

        # CONTOURS
        rects, hulls, src_mask = detect.get_contours(src)

        # CROP
        outs, masks, green = detect.cropping_frame(frame, hulls, src_mask)

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

                roi = cv2.drawContours(roi, hulls, idx, (255, 255, 255), -1)
                roi[roi == 255] = frame[roi == 255]

                if len(corners) == 4:
                    local_orientation = rectify.determineOrientation(i)
                else:
                    local_orientation = sx

                # RECTIFICATION
                text, room, warped, score = rectify.detectKeyPoints(outs[idx], local_orientation)

                if room != "0":
                    tmp = room

                if len(corners) == 4 and score > 260:
                    p = rectify.order_corners(corners)
                    if p != 0:
                        ret = rectify.rectify_image(out_imm_pad.shape[0], out_imm_pad.shape[1], out_imm_pad, p)
                        if not np.isscalar(ret):
                            warped = ret
                            text, room, warped, score = rectify.detectKeyPoints(warped, local_orientation)
                            if room != "0":
                                tmp = room

                if not np.isscalar(warped):
                    im_warped = Image.fromarray(warped)
                    vec_warped = detect.get_feature_vector(im_warped, scaler, to_tensor, normalize, layer, resnet18)

                    prediction_warped = prediction.check(SVM, vec_warped)
                    if prediction_warped:
                        if len(res) >= 3:
                            res.pop(0)
                        res.append({"before": outs[idx], "after": warped})
                        utils.write_local(text, n_frame, n_quadro, warped)

                dict.append({'texts': text, 'rects': rects[idx]})

        dict = utils.check_dict(dict)

        # PERSON
        dict = yolo.detect_person(frame, frame_height, frame_width, dict)

        # RESULT
        for di in dict:
            utils.drawLabel(di['rects'][2], di['rects'][3], di['rects'][0], di['rects'][1], di['texts'], frame)

        frame = utils.resize_output(frame)
        roi = utils.resize_output(roi)

        display = utils.display(tmp, 1080, 1920, frame, roi, res)

        print("--> processed frame number :" + str(n_frame) + "/" + str(frame_length))

        k = cv2.waitKey(5) & 0xFF
        if k == ord("q"):
            break

        # Write the frame into the file 'output.avi'
        out.write(display)

        cv2.imshow("PREVIEW", display)


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
