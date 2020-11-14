import cv2

video = cv2.VideoCapture(0)

check, frame = video.read()

print(check)
print(frame)

cv2.imshow('Capturing', frame)

cv2.waitKey(3000)

video.release()
