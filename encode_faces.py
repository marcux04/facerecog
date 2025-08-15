import face_recognition
import pickle
import cv2
import os

dataset_path = r"C:\Users\anton\OneDrive\Documentos\face_proyect\dataset"
encodings_path = "encodings.pkl"

known_encodings = []
known_names = []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue

    print(f"[INFO] Procesando imágenes de: {person_name}")

    for image_file in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"[WARNING] No se pudo cargar {image_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model="hog")
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

print("[INFO] Guardando encodings...")
data = {"encodings": known_encodings, "names": known_names}
with open(encodings_path, "wb") as f:
    pickle.dump(data, f)

print(f"[INFO] Encodings guardados en: {encodings_path}")
