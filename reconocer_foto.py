import face_recognition
import cv2
import pickle


with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)


image_path = r"C:\Users\anton\Downloads\IMG_20250814_171432.jpg"  
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


boxes = face_recognition.face_locations(rgb, model="hog")
encodings = face_recognition.face_encodings(rgb, boxes)

for (box, encoding) in zip(boxes, encodings):
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Desconocido"

    if True in matches:
        matched_idxs = [i for (i, b) in enumerate(matches) if b]
        counts = {}

        for i in matched_idxs:
            name = data["names"][i]
            counts[name] = counts.get(name, 0) + 1

        name = max(counts, key=counts.get)

    top, right, bottom, left = box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

cv2.imshow("Resultado", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
