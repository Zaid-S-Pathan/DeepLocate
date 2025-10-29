# import face_recognition
# import cv2
# import os
# import glob
# import numpy as np

# class SimpleFacerec:
#     def __init__(self):
#         self.known_face_encodings = []
#         self.known_face_names = []

#         # Resize frame for a faster speed
#         self.frame_resizing = 0.25

#     def load_encoding_images(self, images_path):
#         """
#         Load encoding images from path
#         :param images_path:
#         :return:
#         """
#         # Load Images
#         images_path = glob.glob(os.path.join(images_path, "*.*"))

#         print("{} encoding images found.".format(len(images_path)))

#         # Store image encoding and names
#         for img_path in images_path:
#             img = cv2.imread(img_path)
#             rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#             # Get the filename only from the initial file path.
#             basename = os.path.basename(img_path)
#             (filename, ext) = os.path.splitext(basename)
#             # Get encoding
#             img_encoding = face_recognition.face_encodings(rgb_img)[0]

#             # Store file name and file encoding
#             self.known_face_encodings.append(img_encoding)
#             self.known_face_names.append(filename)
#         print("Encoding images loaded")

#     def detect_known_faces(self, frame):
#         small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
#         # Find all the faces and face encodings in the current frame of video
#         # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#         rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
#         face_locations = face_recognition.face_locations(rgb_small_frame)
#         face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

#         face_names = []
#         for face_encoding in face_encodings:
#             # See if the face is a match for the known face(s)
#             matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
#             name = "Unknown"

#             # # If a match was found in known_face_encodings, just use the first one.
#             # if True in matches:
#             #     first_match_index = matches.index(True)
#             #     name = known_face_names[first_match_index]

#             # Or instead, use the known face with the smallest distance to the new face
#             face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index]:
#                 name = self.known_face_names[best_match_index]
#             face_names.append(name)

#         # Convert to numpy array to adjust coordinates with frame resizing quickly
#         face_locations = np.array(face_locations)
#         face_locations = face_locations / self.frame_resizing
#         return face_locations.astype(int), face_names


import face_recognition
import cv2
import os
import glob
import numpy as np

class SimpleFacerec:
    def __init__(self, camera_index=1):  # Default to Iriun webcam (index 1)
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for faster processing
        self.frame_resizing = 0.25
        
        # Set camera index for Iriun webcam or any other connected webcam
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(self.camera_index)  # Open the webcam

        if not self.cap.isOpened():
            print(f"[ERROR] Could not open camera at index {self.camera_index}.")
            exit()

    def load_encoding_images(self, images_path):
        """
        Load face encodings from the given directory.
        :param images_path: Path to folder containing known images
        """
        images_path = glob.glob(os.path.join(images_path, "*.*"))
        print(f"[INFO] {len(images_path)} encoding images found.")

        for img_path in images_path:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[WARNING] Unable to load image: {img_path}")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            try:
                # Detect faces and encode them
                encodings = face_recognition.face_encodings(rgb_img)
            except Exception as e:
                print(f"[ERROR] Error in face encoding for image {img_path}: {e}")
                continue

            if not encodings:
                print(f"[WARNING] No face found in image: {img_path}")
                continue

            encoding = encodings[0]
            basename = os.path.basename(img_path)
            filename, _ = os.path.splitext(basename)

            self.known_face_encodings.append(encoding)
            self.known_face_names.append(filename)

        print("[INFO] Encoding images loaded successfully.")

    def detect_known_faces(self, frame):
        """
        Detect and recognize faces in a frame.
        :param frame: Frame from webcam
        :return: Tuple of (face_locations, face_names)
        """
        if frame is None:
            print("[ERROR] Received an empty frame.")
            return [], []

        # Resize frame to speed up processing
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Scale face locations back to original frame size
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing

        return face_locations.astype(int), face_names

    def start_webcam_stream(self):
        """
        Start capturing from webcam (Iriun or others).
        """
        if not self.cap.isOpened():
            print("[ERROR] Unable to access webcam.")
            return

        print("[INFO] Starting webcam feed...")
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("⚠️ Could not read frame from webcam.")
                break

            # Detect faces
            face_locations, face_names = self.detect_known_faces(frame)

            # Draw rectangles around faces and label them
            for face_loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = face_loc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

            # Show the frame with recognized faces
            cv2.imshow("Webcam Stream - Press 'q' to Exit", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()


