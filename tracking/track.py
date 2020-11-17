import numpy as np
import torch

from multitracker import JDETracker
from timer import Timer
import visualization as vis


import functions
import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
import pickle
import os


def main(save_detected_videos=False):
    # TODO load in dataset here

    video_names = ['KevinTester.mp4']  # List of video names

    """
    The ground-truth detections with format {video_name: [(face_locations, face_names), ...]} where the list is of length T
    face_locations is a list of length N with format [(top, right, bottom, left), ...]
    face_names is a list of length N with format [name, ...]
    """
    ground_truth_detections = {}

    # TODO load in pre-computed face encodings from our training set
    known_face_dict = pickle.load(open('face_encodings_fixed1.pickle', 'rb'))
    known_face_dict['features'] = [x[0] for i, x in enumerate(known_face_dict['features'])]
    known_face_dict['labels'] = [(x + ' ' + str(i)) for i, x in enumerate(known_face_dict['labels'])]
    print(len(known_face_dict['features'][0]))

    detections = {}
    for video_name in video_names:
        cap = cv2.VideoCapture(video_name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        tracker = JDETracker(frame_rate=fps)
        timer = Timer()

        if save_detected_videos:
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))

            out_vid_name = video_name.split('.')[0] + '_detected2.avi'
            size = (frame_width, frame_height)
            out_video = cv2.VideoWriter(out_vid_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

        dets = []
        frame_id = 0
        while cap.isOpened():
            timer.tic()
            ret, frame = cap.read()

            if frame is None:
                break

            frame = frame[:, :, ::-1]
            online_targets = tracker.update(frame)
            online_tlwhs = []
            online_ids = []

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                online_tlwhs.append(tlwh)
                online_ids.append(tid)

            timer.toc()

            # save results
            dets.append((frame_id + 1, online_tlwhs, online_ids))
            print('frame: {} fps: {:.2} dets: {}'.format(frame_id, 1. / timer.average_time, online_tlwhs))
            if save_detected_videos:
                frame = vis.plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id,
                                          fps=1. / timer.average_time)
                #cv2.imshow('vid', frame)
                frame = frame[:, :, ::-1]
                out_video.write(frame)

            frame_id += 1

        detections[video_name] = dets

        cap.release()

        if save_detected_videos:
            out_video.release()


if __name__ == '__main__':
    main(True)
