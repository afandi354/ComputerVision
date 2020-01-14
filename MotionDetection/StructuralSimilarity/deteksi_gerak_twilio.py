#Menggunakan python3
import cv2
from skimage.measure import compare_ssim
from twilio.rest import Client
from twilio_creds import ACCOUNT_SID, AUTH_TOKEN, TWILIO_PHONE
from phone_number import PHONE_NUMBER

def ssim(A, B):
    return compare_ssim(A, B, data_range=A.max()-A.min())

# Inisialisasi Twilio client
client = Client(ACCOUNT_SID, AUTH_TOKEN)

# open webcam (import video)
video_capture = cv2.VideoCapture('video.mp4')
# webcam: video_capture = cv2.VideoCapture(0)

# membaca frame pertama
_, current_frame = video_capture.read()

# konversi ke grayscale
current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

# inisialisasi frame lainnya
previous_frame = current_frame

frame_counter = 1
is_first_message = True

# main loop
while True:
    # membaca frame berikutnya (dan konversi ke grayscale)
    _, current_frame = video_capture.read()
    if current_frame is None:
        break
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    if frame_counter % 10 == 0:
        # membandingkan dua frames
        ssim_index = ssim(current_frame, previous_frame)
        if ssim_index < 0.90 and is_first_message:
            # mengirim pesan melalui Twilio
            client.messages.create(body='Seseorang telah memasuki ruangan!', from_=TWILIO_PHONE, to=PHONE_NUMBER)
            print('Intruder Alert!')
            is_first_message = False

        # update frame sebelumnya
        previous_frame = current_frame

    # menampilkan video/webcam
    cv2.imshow('app', current_frame)
    cv2.waitKey(1)
    frame_counter += 1

video_capture.release()
cv2.destroyAllWindows()
