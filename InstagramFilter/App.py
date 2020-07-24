import cv2
import numpy as np

def dummy(value):
    pass

#inisialisasi kernel
identity_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
box_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], np.float32)/9.0
# 1 + 2 + 3 + 4 + 5 / 5
gaussian_kernel1 = cv2.getGaussianKernel(3, 0)
gaussian_kernel2 = cv2.getGaussianKernel(5, 0)

kernels = [identity_kernel, sharpen_kernel, box_kernel, gaussian_kernel1, gaussian_kernel2]

# Membaca citra asli & convert ke grayscale
citra_asli = cv2.imread('auba.jpg')
gray_convert = cv2.cvtColor(citra_asli, cv2.COLOR_BGR2GRAY)

# Membuat tampilan UI (Window dan Trackbar)
cv2.namedWindow('Aplikasi Filter Instagram')
# argument : NamaTrackbar, NamaWindow, NilaiAwal, NilaiMax, onChange (event handle)
cv2.createTrackbar('contrast', 'Aplikasi Filter Instagram', 1, 100, dummy)

# nama : brighness, nilai awal = 50, nilai max= 100, event handler
cv2.createTrackbar('brightness', 'Aplikasi Filter Instagram', 50, 100, dummy)
cv2.createTrackbar('filter', 'Aplikasi Filter Instagram', 0, len(kernels)-1, dummy) #update nilai max dari filter
cv2.createTrackbar('grayscale', 'Aplikasi Filter Instagram', 0, 1, dummy)

# Main UI Loop
count = 1
while True :
    # TODO : membaca semua nilai trackbar
    grayscale = cv2.getTrackbarPos('grayscale','Aplikasi Filter Instagram')
    contrast = cv2.getTrackbarPos('contrast', 'Aplikasi Filter Instagram')
    brightness = cv2.getTrackbarPos('brightness', 'Aplikasi Filter Instagram')
    kernel_idx = cv2.getTrackbarPos('filter', 'Aplikasi Filter Instagram')
    # TODO : apply the filters
    color = cv2.filter2D(citra_asli, -1, kernels[kernel_idx])
    gray = cv2.filter2D(gray_convert, -1, kernels[kernel_idx])
    # TODO : apply the brightnes and contrast
    color = cv2.addWeighted(color, contrast, np.zeros_like(citra_asli), 0, brightness - 50)
    gray = cv2.addWeighted(gray, contrast, np.zeros_like(gray_convert), 0, brightness - 50)

    # menambah fungsi untuk keluar dan simpan gambar
    key  = cv2.waitKey(100)
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save image
        if grayscale == 0:
            cv2.imwrite('output_{}.png'.format(count), color)
        else:
            cv2.imwrite('output_{}.png'.format(count), gray)
        count += 1

    # Menampilkan Gambar
    if grayscale == 0:
        cv2.imshow('Aplikasi Filter Instagram', color)
    else:
        cv2.imshow('Aplikasi Filter Instagram', gray)

#untuk menutup semua windows
cv2.destroyAllWindows()