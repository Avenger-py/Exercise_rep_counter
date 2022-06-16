#output video file link: https://drive.google.com/file/d/1PwYrnQY3bEKGz-VWdauMiOGocjPfwU3a/view?usp=sharing

import cv2
import mediapipe as mp
import numpy as np
from statistics import mean
import time

#input video
cap = cv2.VideoCapture('KneeBend.mp4')

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
#print(frame_height, frame_width)

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calc_angle(a, b, c):
    #calculate angle at b
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    deg = np.abs(180*rad/np.pi)

    if deg > 180.0:
        deg = 360 - deg
    return deg


#params
rep_count = 0
flag = None
timer = 'stop'
angles = []
angle_threshold = 80
time_threshold = 8

#video writer
output = cv2.VideoWriter('KneeBend_output_ShantanuSharma.mp4', cv2.VideoWriter_fourcc(*'MJPG'),
                         25, (frame_width, frame_height))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, frame = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    frame.flags.writeable = False
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame)
    #print(results)

    # Draw the pose annotation on the image.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    try:
        lndmrks = results.pose_landmarks.landmark
        #print(lndmrks)

        left_hip = [lndmrks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    lndmrks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

        left_knee = [lndmrks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                     lndmrks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]

        left_ankle = [lndmrks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                      lndmrks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

        angle = round(calc_angle(left_hip, left_knee, left_ankle), 0)

        angles.append(angle)

        #averaging last 3 angle values for better stability
        if len(angles) < 3:
            angle = round(mean(angles), 0)
        else:
            angle = round(mean(angles[-3:]), 0)

        #cv2.putText(frame, str(angle), tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                    #cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if angle < angle_threshold:
            flag = 'bent'
            if timer == 'stop':
                start = time.time()
                timer = 'start'
            stop = time.time()
            t = int(stop - start)
            #print(f'start = {start}, stop =  {stop}, t = {t}, timer = {timer}')
            if t>time_threshold:
                msg = 'Straighten your leg'
                cv2.putText(frame, msg, (0, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            cv2.putText(frame, 'timer = ' + str(t), (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        if angle > angle_threshold and flag == 'bent':
            if t >= time_threshold:
                rep_count += 1
                flag = 'not bent'
                timer = 'stop'
                #print(msg)
            else:
                msg = 'Keep your knee bent'
                cv2.putText(frame, msg, (0, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                timer = 'stop'
                #print(msg)

        cv2.putText(frame, 'rep count = ' + str(rep_count), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    except:
        print('No landmarks found !!')
        pass

    output.write(frame)
    cv2.imshow('output', frame)
    k = cv2.waitKey(10) & 0xFF
    if k == ord('q'):
        break

print(f'Total reps = {rep_count}')
cap.release()
output.release()