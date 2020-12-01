import cv2
import pickle
from multitracker import JDETracker
from timer import Timer
import visualization as vis


def main(save_detected_videos=False):
    video_names = ['vid2.mp4']

    """
    The ground-truth detections with format {video_name: [(face_locations, face_names), ...]} where the list is of length T
    face_locations is a list of length N with format [(top, right, bottom, left), ...]
    face_names is a list of length N with format [name, ...]
    """
    ground_truth_detections = {}

    known_face_dict = pickle.load(open('face_encodings_final.pickle', 'rb'))
    known_face_dict['features'] = [x[0] for i, x in enumerate(known_face_dict['features'])]
    known_face_dict['labels'] = [x for i, x in enumerate(known_face_dict['labels'])]
    print('number of known faces: ', len(known_face_dict['features'][0]))

    detections = {}
    for vid_name in video_names:
        cap = cv2.VideoCapture(vid_name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        tracker = JDETracker(known_face_dict, frame_rate=fps)
        timer = Timer()

        if save_detected_videos:
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            print(frame_width, frame_height)
            out_vid_name = vid_name.split('.')[0] + '_sort_results.avi'
            size = (frame_width, frame_height)
            out_video = cv2.VideoWriter(out_vid_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error opening video stream or file")
            pass

        dets = []
        frame_id = 0

        # Read until video is completed
        while cap.isOpened():
            # Capture frame-by-frame
            if frame_id > 0:
                timer.tic()
            ret, frame = cap.read()
            if ret:
                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

                frame = frame[:, :, ::-1]
                targets = tracker.update(frame)
                
                duration = 0.0000001
                if frame_id > 0:
                    duration = timer.toc(average=False)

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
                    # cv2.imshow(vid_name, frame)
                    out_video.write(frame)

                frame_id += 1

            else:
                break

        detections[vid_name] = dets
        cap.release()
        if save_detected_videos:
            out_video.release()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main(True)
