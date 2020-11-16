import face_recognition
import time
from scipy.misc import imresize
import cv2
import numpy as np


def get_cropped_faces(image, resize=None):
    """
    :param image: An image of shape (H, W, 3)
    :param resize: Optional resize parameter (h, w) to resize the cropped face.
    
    :return: Returns a tuple of lists:
             1) The cropped faces which have been resized to the specified shape [(h, w), ...]. 
             2) The bounding box locations of these faces [(top, right, bottom, left), ...]. 
    """
    
    face_locations = face_recognition.face_locations(image)  # (top, right, bottom, left)
    
    cropped_faces = []
    for bbox in face_locations:
        crop = image[bbox[0]:bbox[2], bbox[3]:bbox[1]]
        if resize is not None:
            crop = imresize(crop, resize)
        cropped_faces.append(crop)
        
    return cropped_faces, face_locations


def extract_features_image(image):
    """
    
    :param image: An image of shape (H, W, 3)
    
    :return: Returns a list of face features for the faces in the image [[f1, ...], ...]
    """

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    return face_encodings
    
    
def extract_features_video(video):
    """
    
    :param image: An image of shape (T, H, W, 3)
    
    :return: Returns a dictionary of face features for the faces in the image with format {frame_id: feature_list}
             
             frame_id: the frame number in the range [0, T-1]
             feature_list: The list of features [[f1, ...], ...]
    """
    output_dict = {}
    
    for i in range(video.shape[0]):
        face_locations = face_recognition.face_locations(video[i])
        face_encodings = face_recognition.face_encodings(video[i], face_locations)
        
        output_dict[i] = face_encodings
    
    return output_dict


def detect_faces(image, feature_dict, match_threshold=0.6, resize_factor=1.0):
    """
    
    :param image: An image of shape (H, W, 3)
    :param feature_dict: A dictionary containing all reference features and their corresponding labels {'features': [...], 'labels': [...]}
    :param resize_factor: A float value in range (0.0, 1.0] which denotes how much the image will be resized in each dimension.
    
    :return: Returns a tuple of lists:
             1) The bounding box locations of the detected faces [(top, right, bottom, left), ...].
             2) The predicted label for each detection ['Name', ...]
    """
    
    assert resize_factor <= 1.0
    
    if resize_factor < 1.0:
        image = imresize(image, resize_factor)

    #start_time = time.time()
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    #print(time.time() - start_time)
    
    if len(face_locations) == 0:
        return [], []
    
    known_face_encodings = feature_dict['features']
    known_face_names = feature_dict['labels']
    
    face_names = []
    for face_encoding in face_encodings:
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    
        best_match_index = np.argmin(face_distances)
    
        if face_distances[best_match_index] <= match_threshold:
            face_names.append(known_face_names[best_match_index])
        else:
            face_names.append('Unknown')
    
    if resize_factor < 1.0:
        up = 1.0/resize_factor
        face_locations = [(top*up, right*up, bottom*up, left*up) for (top, right, bottom, left) in face_locations]
    
    return face_locations, face_names
    

def detect_faces_video(video, feature_dict, match_threshold=0.6, resize_factor=1.0):
    """
    
    :param image: A video of shape (T, H, W, 3)
    :param feature_dict: A dictionary containing all reference features and their corresponding labels {'features': [...], 'labels': [...]}
    :param resize_factor: A float value in range (0.0, 1.0] which denotes how much the each frame will be resized in each dimension.
    
    :return: Returns a dictionary containing tuple of lists for each frame of format {frame_id: (boxes, labels), ...}
              
             frame_id: the frame number in the range [0, T-1]
             boxes: The bounding box locations of the detected faces [(top, right, bottom, left), ...]
             labels: The predicted label for each detection ['Name', ...]
    """
    
    output_dict = {}
    
    for i in range(video.shape[0]):
        output_dict[i] = detect_faces(video[i], feature_dict, match_threshold)

    return output_dict
    
    
def display_results(image, face_locations, face_names):
    """
    :param image: An image of shape (H, W, 3)
    :param face_locations: The bounding box locations of the detected faces [(top, right, bottom, left), ...]
    :param face_names: The predicted label for each detected face ['Name', ...]
    
    :return: Returns an image with the bounding boxes/names "drawn" on. May alter the input image.
    """
    image = np.ascontiguousarray(image, dtype=np.uint8)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top, right, bottom, left = int(top), int(right), int(bottom), int(left)
        # Draw a box around the face
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    return image
    
    
    
