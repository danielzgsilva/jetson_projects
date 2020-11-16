import functions
import numpy as np
import cv2
import face_recognition
import matplotlib.pyplot as plt
import pickle
import os


def evaluate_detections(gt_detections, predicted_detections):

    return 0


def test_on_dataset(save_detected_videos=False):
    
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
    known_face_dict['labels'] = [(x+ ' ' + str(i)) for i, x in enumerate(known_face_dict['labels'])]
    print(len(known_face_dict['features'][0]))
    
    #print(known_face_dict['labels'])
    #exit()
    """
    known_face_dict = {'features': [], 'labels': []}
    
    for img_name in os.listdir('./KevinFaces/'):

        image = cv2.imread('./KevinFaces/'+img_name)
        image = image[:, :, ::-1]
        features = functions.extract_features_image(image)

        if len(features) == 0:
            print('No Detection', img_name)
            continue
        elif len(features) > 1:
            print('Many Detections', img_name)
            continue
        
        known_face_dict['features'].append(features[0])
        known_face_dict['labels'].append('Kevin %d' % (len(known_face_dict['features'])))
    """
    detections = {}
    for video_name in video_names:
        cap = cv2.VideoCapture(video_name)
        
        if save_detected_videos:
            frame_width = int(cap.get(3)) 
            frame_height = int(cap.get(4))
            
            out_vid_name = video_name.split('.')[0] + '_detected2.avi'
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            size = (frame_width, frame_height)
            out_video = cv2.VideoWriter(out_vid_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)

        dets = []
        while(cap.isOpened()):
            ret, frame = cap.read()

            if frame is None:
                break
            
            frame = frame[:, :, ::-1]
            
            face_locations, face_names = functions.detect_faces(frame, known_face_dict, resize_factor=0.1)
            
            dets.append((face_locations, face_names))
            
            if save_detected_videos:
                print(frame.shape, face_locations)
                frame = functions.display_results(frame, face_locations, face_names)
                frame = frame[:, :, ::-1]
                out_video.write(frame)

        detections[video_name] = dets
        
        cap.release()
        
        if save_detected_videos:
            out_video.release()

    evaluate_detections(ground_truth_detections, detections)


def test_on_stream():
    # TODO load in pre-computed face encodings from our training set
    
    known_face_dict = {'features': [], 'labels': []}

    video_capture = cv2.VideoCapture(0)
    
    while True:
        ret, frame = video_capture.read()
        
        frame = frame[:, :, ::-1]
            
        face_locations, face_names = functions.detect_faces(frame, known_face_dict, resize_factor=1.0)
        
        frame = functions.display_results(frame, face_locations, face_names)
        
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_on_dataset(True)
    #exit()
    
    """
    test_image = face_recognition.load_image_file('group_photo.jpeg')
    train_image = face_recognition.load_image_file('reu2018.jpg')
    
    extracted_features = functions.extract_features_image(train_image)
    print(len(extracted_features))
    labels = ['Person %d' % i for i in range(len(extracted_features))]
    
    known_face_dict = {'features': extracted_features, 'labels': labels}
    
    face_locations, face_names = functions.detect_faces(train_image, known_face_dict, resize_factor=1.0, match_threshold=0.5)
    frame1 = functions.display_results(train_image, face_locations, face_names)
    
    face_locations, face_names = functions.detect_faces(test_image, known_face_dict, resize_factor=1.0, match_threshold=0.5)
    frame2 = functions.display_results(test_image, face_locations, face_names)
    
    plt.imshow(frame1)
    plt.show()
    plt.imshow(frame2)
    plt.show()
    """
    
