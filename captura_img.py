import cv2
import os

name = "chris evans"
student_id = "utm"
output_dir = f"dataset/{name}_{student_id}"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while count < 100:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capturando rostro", frame)

   
    key = cv2.waitKey(1)
    if key == ord('s'):
        img_path = os.path.join(output_dir, f"{name}_{student_id}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"Imagen guardada: {img_path}")
        count += 1

    
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
