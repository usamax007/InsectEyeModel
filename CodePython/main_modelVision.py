# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 18:22:45 2024

@author: Usama
"""

from compoundEyeModel import CompoundEyeModel
import cv2
import numpy as np
#%%
output_frame_width = 186
output_frame_height = 186

#%%
video_path = "flower_scn.mp4"  # Replace with your video file path
fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
video_capture = cv2.VideoCapture(video_path)
video_vision = cv2.VideoWriter('output_vis.mp4', fourcc, 30, (output_frame_width, output_frame_width),isColor=False)
video_eventp = cv2.VideoWriter('output_epm.mp4', fourcc, 30, (output_frame_width, output_frame_width),isColor=False)

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

eye_model = CompoundEyeModel('Ms',180,180) # 77 and 35 by default
frame_number = 0
while True:
    print(f"frame number {frame_number}")
    ret, frame_rgb = video_capture.read()
    if not ret or frame_number == 25:
        break
    frame_gsc = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
    matrixVisualField = eye_model.giveVisualFieldMatrix(frame_gsc,(8.26e-5)/12,(8.26e-5)/12)
    # matrixVisualField = frame_gsc
    if not video_vision.write(cv2.resize(matrixVisualField, (output_frame_width,output_frame_height), interpolation=cv2.INTER_LINEAR)):
        print("error writing to output file output_vis.avi")
    if frame_number > 0:
        matrixEventPolarity = eye_model.giveEventPolarityMatrix(True)
        # matrixEventPolarity = frame_gsc
        if not video_eventp.write(cv2.resize(matrixEventPolarity, (output_frame_width,output_frame_height), interpolation=cv2.INTER_LINEAR)):
            print("error writing to output file output_epm.avi")
        
    # cv2.imshow('Frame', matrixEventPolarity)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #   break

    #cv2.imwrite(f"frame_{frame_number}.jpg", frame)
    frame_number += 1

video_capture.release()
video_vision.release()
video_eventp.release()
cv2.destroyAllWindows()

#%%
