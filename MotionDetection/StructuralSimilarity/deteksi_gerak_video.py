import cv2
from skimage.measure import compare_ssim

def ssim(A, B):
    return compare_ssim(A, B, data_range=A.max()-A.min())

#membaca file video
video_capture = cv2.VideoCapture('video.mp4')

#membaca frame pertama
_, current_frame = video_capture.read()

#convert to grayscale
current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

#inisialisasi frame lain
previous_frame = current_frame

frame_counter = 1

#looping
while True:
    #baca frame berikutnya (dan convert ke Grayscale)
    _, current_frame = video_capture.read()
    if current_frame is None:
        break
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    if frame_counter % 20 == 0:
        #membandingkan dua frame
        ssim_index = ssim(current_frame, previous_frame)
        if ssim_index < 0.9:
            print('Penyusup ditemukan!')

    #update frame sebelumnya
    previous_frame = current_frame

    #menampilkan video
    cv2.imshow('VideoKu', current_frame)
    cv2.waitKey(10)
    frame_counter += 1

video_capture.release()
cv2.destroyAllWindows()
