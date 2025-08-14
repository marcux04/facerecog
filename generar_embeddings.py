import face_recognition
import os
import cv2
import pickle


dataset_dir = r"C:\Users\anton\OneDrive\Documentos\face_proyect\dataset"
known_encodings = []
known_names = []


for person_name in os.listdir(dataset_dir):
    person_path = os.path.join(dataset_dir, person_name)

    if not os.path.isdir(person_path):
        continue

    for filename in os.listdir(person_path):
        image_path = os.path.join(person_path, filename)
        image = cv2.imread(image_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)


data = {"encodings": known_encodings, "names": known_names}
with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("[INFO] Embeddings generados y guardados.")
