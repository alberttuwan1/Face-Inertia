import cv2
import sys
from utils import *
from yunet import YuNet
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
import pygame

EAR_THRESHOLD = 0.26
MAR_THRESHOLD = 0.5

pygame.mixer.init()
pygame.mixer.music.load("./Alarm/alarm1.mp3")

def predictLandmarks(predictor, faces):
    landmarksArray = []
    for (bbox, conf, frame) in faces:
        x, y, w, h = bbox
        landmarks = predictor.process(frame[y:y + h, x:x + w]).multi_face_landmarks
        landmarksArray.append([bbox, conf, landmarks])
    return landmarksArray


def findFaces(finder, frame):
    results = finder.infer(frame)
    faces = []
    for det in results:
        bbox = det[0:4].astype(np.int32)
        conf = det[-1]
        faces.append((bbox, conf, frame))
    return faces


def getMouthRatio(landmarksArray):
    mouth_pairs = []
    for (bbox, conf, landmarks) in landmarksArray:
        if landmarks is None:
            return 0
        for landmark in landmarks:
            for i in MOUTH_TOP_3:
                first = i[0]
                second = i[1]
                t = landmark.landmark[first]
                b = landmark.landmark[second]
                mouth_pairs.append(dist.euclidean((int(t.x * bbox[2]), int(t.y * bbox[3])),
                                                  (int(b.x * bbox[2]), int(b.y * bbox[3]))))
    MAR = (mouth_pairs[0] + mouth_pairs[1] + mouth_pairs[2]) / (3.0 * mouth_pairs[3])
    return MAR


def getEyeRatio(landmarksArray):
    left_eye_pairs = []
    right_eye_pairs = []
    for (bbox, conf, landmarks) in landmarksArray:
        if landmarks is None:
            return 1
        for landmark in landmarks:
            for i in LEFT_EYE_TOP_3:
                first = i[0]
                second = i[1]
                t = landmark.landmark[first]
                b = landmark.landmark[second]
                left_eye_pairs.append(dist.euclidean((int(t.x * bbox[2]), int(t.y * bbox[3])),
                                                     (int(b.x * bbox[2]), int(b.y * bbox[3]))))

            for i in RIGHT_EYE_TOP_3:
                first = i[0]
                second = i[1]
                t = landmark.landmark[first]
                b = landmark.landmark[second]
                right_eye_pairs.append(dist.euclidean((int(t.x * bbox[2]), int(t.y * bbox[3])),
                                                      (int(b.x * bbox[2]), int(b.y * bbox[3]))))

    leftEAR = (left_eye_pairs[0] + left_eye_pairs[1] + left_eye_pairs[2]) / (3.0 * left_eye_pairs[3])
    rightEAR = (right_eye_pairs[0] + right_eye_pairs[1] + right_eye_pairs[2]) / (3.0 * right_eye_pairs[3])
    return (leftEAR + rightEAR) / 2.0


def visualize(image, landmarksArray, EAR, MAR, yawningCount, blinkCount, isYawning, eyeUnfocusedTime, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (0 + 240, 0 + 110), (0, 0, 0, 0), -1)
    cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
    cv2.putText(output, 'EAR: {:.2f}'.format(EAR), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
    cv2.putText(output, 'MAR: {:.2f}'.format(MAR), (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
    cv2.putText(output, f"YAWNING: {str(isYawning).upper()}", (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
    cv2.putText(output, 'BLINK COUNT: {:.2f}'.format(blinkCount), (0, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
    cv2.putText(output, 'YAWNING COUNT: {:.2f}'.format(yawningCount), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)
    cv2.putText(output, 'EYE UNFOCUSED TIME: {:.2f}s'.format(eyeUnfocusedTime), (0, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for (bbox, conf, landmarks) in landmarksArray:
        cv2.rectangle(output, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), box_color, 2)
        cv2.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1] + 12), cv2.FONT_HERSHEY_DUPLEX, 0.5, text_color)
        if landmarks is None:
            break
        for landmark in landmarks:
            for i in ROI_LANDMARKS:
                l = landmark.landmark[i]
                cv2.circle(output, (int(l.x * bbox[2]) + bbox[0], int(l.y * bbox[3]) + bbox[1]), 1, (0, 0, 255), -1)
    return output

def play_alarm():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play(-1)

def stop_alarm():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()


def main() -> int:
    finder = YuNet('./weights/face_detection_yunet_2023mar.onnx')
    predictor = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.1)
    captureDevice = cv2.VideoCapture(1)
    captureDevice.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    w = int(captureDevice.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(captureDevice.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finder.setInputSize([w, h])

    tickMeter = cv2.TickMeter()
    eyeUnfocusedMeter = cv2.TickMeter()
    yawningMeter = cv2.TickMeter()
    blinkCount = 0
    eyeClosed = False
    eyeUnfocusedTime = 0
    startTicks = 0
    isYawning = False
    yawningCount = 0
    currentAlarm = None  # Variable to track the currently playing alarm ("yawn", "eye", or None)

    eyeUnfocusedMeter.start()
    while cv2.waitKey(1) < 0:
        hasFrame, frame = captureDevice.read()
        if hasFrame:
            tickMeter.start()
            faces = findFaces(finder, frame)

            if len(faces) >= 1:
                landmarksArray = predictLandmarks(predictor, faces)
                EAR = getEyeRatio(landmarksArray)
                MAR = getMouthRatio(landmarksArray)

                # Eye closure logic
                if EAR < EAR_THRESHOLD:
                    if not eyeClosed:
                        startTicks = cv2.getTickCount()
                        eyeUnfocusedMeter.start()
                        eyeClosed = True
                    else:
                        eyeUnfocusedTime = (cv2.getTickCount() - startTicks) / cv2.getTickFrequency()
                else:
                    eyeUnfocusedMeter.stop()
                    if eyeUnfocusedMeter.getTimeMilli() >= 100:
                        blinkCount += 1
                    eyeUnfocusedMeter.reset()
                    eyeClosed = False
                    eyeUnfocusedTime = 0

                # Yawning logic
                if MAR > MAR_THRESHOLD:
                    if not isYawning:
                        yawningMeter.reset()
                        yawningMeter.start()
                        isYawning = True

                    if currentAlarm is None:
                        currentAlarm = "yawn"
                        play_alarm()

                    cv2.putText(frame, "YAWNING DETECTED!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    if isYawning:
                        yawningMeter.stop()
                        if yawningMeter.getTimeMilli() >= 3000:
                            yawningCount += 1
                        yawningMeter.reset()
                    isYawning = False

                    if currentAlarm == "yawn":
                        currentAlarm = None
                        stop_alarm()

                # Alarm for prolonged eye closure
                if eyeUnfocusedTime >= 2:
                    currentAlarm = "eye"
                    play_alarm()
                    cv2.putText(frame, "EYES CLOSED TOO LONG!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    if currentAlarm == "eye":
                        pygame.mixer.music.stop()
                        currentAlarm = None

                tickMeter.stop()

                # Visualization
                frame = visualize(
                    frame, landmarksArray, EAR, MAR, yawningCount, blinkCount,
                    isYawning, eyeUnfocusedTime,
                    text_color=(0, 0, 255) if (isYawning or eyeUnfocusedTime >= 2) else (0, 255, 0),
                    fps=tickMeter.getFPS()
                )

            cv2.imshow('Inertia Output', frame)
            tickMeter.reset()
        else:
            print("[!] Video capture from device failed... aborting!")
            break

    captureDevice.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    sys.exit(main())
