## this file will contain all the important functions we have created in the nitebooks so theyt can be reused

# sfm_utils.py

import cv2
import numpy as np
import os

TARGET_WIDTH = 1280

# ---- Feature extractor ----
def get_feature_extractor():
    try:
        sift = cv2.SIFT_create()
        return sift, False  # not ORB
    except AttributeError:
        orb = cv2.ORB_create(5000)
        return orb, True


def load_gray_image(dataset_dir, filename):
    """
    Load an image from dataset_dir/filename, resize if wide, return grayscale.
    """
    path = os.path.join(dataset_dir, filename)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load {path}")
    h, w = img.shape[:2]
    if w > TARGET_WIDTH:
        scale = TARGET_WIDTH / w
        img = cv2.resize(img, (TARGET_WIDTH, int(h * scale)))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def match_features(img1_gray, img2_gray, ratio_thresh=0.75):
    """
    Detect features in two grayscale images and return keypoints + good matches.
    """
    feature, use_orb = get_feature_extractor()
    if use_orb:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    kp1, des1 = feature.detectAndCompute(img1_gray, None)
    kp2, des2 = feature.detectAndCompute(img2_gray, None)

    knn = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in knn:
        if m.distance < ratio_thresh * n.distance:
            good.append(m)

    return kp1, kp2, good


def build_intrinsics_from_image_size(width, height):
    """
    Build simplified K with fx=fy=width, cx=width/2, cy=height/2.
    """
    fx = fy = float(width)
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return K
