# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 23:11:41 2024

@author: Usama
"""

from compoundEyeModel import CompoundEyeModel
import cv2
import numpy as np
#%%
video_path = "flower_scn_bi.mp4"  # Replace with your video file path
video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    raise IOError("Could not open video file")

fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}")
print(f"Frame Width: {frame_width}")
print(f"Frame Height: {frame_height}")
print(f"Total Frames: {frame_count}")

frame_number = 0
frame_0 = [[]]
frame_1 = [[]]
while True:
    ret, frame = video_capture.read()
    if not ret or frame_number == 22:
        break
    frame_0 = frame_1
    frame_1 = frame
    # cv2.imshow('Frame', frame)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break

    #cv2.imwrite(f"frame_{frame_number}.jpg", frame)
    frame_number += 1

video_capture.release()
cv2.destroyAllWindows()

frame_0 = cv2.cvtColor(frame_0, cv2.COLOR_BGR2GRAY)
cv2.imshow('Frame 0', frame_0)
frame_1 = cv2.cvtColor(frame_1, cv2.COLOR_BGR2GRAY)
cv2.imshow('Frame 1', frame_1)
#%%
eye_model = CompoundEyeModel('Ms',180,180) # 77 and 35 by default
matrixVisualField_0 = eye_model.giveVisualFieldMatrix(frame_0,(8.26e-5)/12,(8.26e-5)/12)
matrixVisualField_1 = eye_model.giveVisualFieldMatrix(frame_1,(8.26e-5)/12,(8.26e-5)/12)
#%%
resized_image_0 = cv2.resize(matrixVisualField_0, (186*5,186*5), interpolation=cv2.INTER_LINEAR)
resized_image_1 = cv2.resize(matrixVisualField_1, (186*5,186*5), interpolation=cv2.INTER_LINEAR)
cv2.imshow('Frame 3', resized_image_0)
cv2.imshow('Frame 4', resized_image_1)
#%%
matrixEventPolarity = eye_model.giveEventPolarityMatrix(True)
resized_image_3 = cv2.resize(matrixEventPolarity, (186*5,186*5), interpolation=cv2.INTER_LINEAR)
cv2.imshow('Frame 5', resized_image_3)
