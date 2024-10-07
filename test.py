import face_recognition
import numpy as np
import cv2
import os

def load_known_faces(known_image_paths):
    known_face_encodings = []
    known_names = []
    for image_path in known_image_paths:
        image = face_recognition.load_image_file(image_path)
        face_encodings = face_recognition.face_encodings(image)
        if len(face_encodings) > 0:
            face_encoding = face_encodings[0]
            known_face_encodings.append(face_encoding)
            name = image_path.split(".")[0].split("/")[-1]
            known_names.append(name)
    return known_face_encodings, known_names

def recognize_faces(frame, known_face_encodings, known_names, distance_threshold=0.6):
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=distance_threshold)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_names[best_match_index]
            top, right, bottom, left = face_location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            name="unknown"
    return frame

if __name__ == "__main__":
    folder_path="known_faces/"
    all_files = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            all_files.append(os.path.join(folder_path, filename))
    known_face_encodings, known_names = load_known_faces(all_files)
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        result_frame = recognize_faces(frame, known_face_encodings, known_names)
        cv2.imshow("Face Recognition", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('z'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
