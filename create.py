import cv2
A=input("Enter the name:")
A=A+'.jpg'
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
else:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture an image.")
    else:
        cv2.imwrite("known_faces/"+A, frame)
        print("Completed!")
    cap.release()
cv2.destroyAllWindows()