import os
import numpy as np
from collections import OrderedDict

LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382, 362, 263]  
RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7] 
MOUTH_LANDMARKS = [0, 267, 269, 270, 409, 306, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]  
ROI_LANDMARKS = LEFT_EYE_LANDMARKS + RIGHT_EYE_LANDMARKS + MOUTH_LANDMARKS

RIGHT_EYE_TOP_3 = [(160, 144), (159, 145), (158, 153), (33, 133)]
LEFT_EYE_TOP_3 = [(385, 380), (386, 374), (387, 373), (362, 263)]
MOUTH_TOP_3 = [(82, 87), (13, 14), (312, 317), (62, 306)]

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

def clearScreen():
    os.system("cls||clear")


def dlib2coords(landmarks, n_landmarks = 68, dtype="int"):
	coords = np.zeros((n_landmarks, 2), dtype=dtype)
	for i in range(0, n_landmarks):
		coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
	return coords