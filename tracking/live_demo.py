import cv2
import pickle
from multitracker import JDETracker
from timer import Timer
import visualization as vis


def main(save_detected_videos=False):
    known_face_dict = pickle.load(open('face_encodings_fixed1.pickle', 'rb'))
    known_face_dict['features'] = [x[0] for i, x in enumerate(known_face_dict['features'])]
    known_face_dict['labels'] = [(x + ' ' + str(i)) for i, x in enumerate(known_face_dict['labels'])]
    print('number of known faces: ', len(known_face_dict['features'][0]))

    cap = cv2.VideoCapture("~/dev/video1")
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    tracker = JDETracker(known_face_dict, frame_rate=fps)
    timer = Timer()

    if save_detected_videos:
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        out_vid_name = 'demo_output.avi'
        size = (frame_width, frame_height)
        out_video = cv2.VideoWriter(out_vid_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream")

    dets = []
    frame_id = 0

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            frame = frame[:, :, ::-1]
            timer.tic()
            targets = tracker.update(frame)
            timer.toc()

            tlwhs = []
            ids = []
            names = []
            for t in targets:
                tlwhs.append(t.tlwh)
                ids.append(t.track_id)
                names.append(t.track_name)

            dets.append((frame_id + 1, tlwhs, ids))
            print('frame: {} fps: {:.2} dets: {}'.format(frame_id, 1. / timer.average_time, tlwhs))

            if save_detected_videos:
                frame = vis.plot_tracking(frame, tlwhs, ids, frame_id=frame_id,
                                          fps=1. / timer.average_time, names=names)
                frame = frame[:, :, ::-1]
                cv2.imshow('Group 4 Live Demo', frame)
                out_video.write(frame)

            frame_id += 1

        else:
            break

    cap.release()
    if save_detected_videos:
        out_video.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(True)