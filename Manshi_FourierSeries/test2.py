from pathlib import Path

import cv2
import numpy as np

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()


def fdmap_channel(channel):
    float_ch = np.float32(channel)
    dft = cv2.dft(float_ch, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(
        cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-6
    )
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def filter_channel(channel, filter):
    float_img = np.float32(channel)
    dft = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    filtered = dft_shift * filter[:, :, np.newaxis]
    idft_shift = np.fft.ifftshift(filtered)
    img_back = cv2.idft(idft_shift, flags=cv2.DFT_REAL_OUTPUT)
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def butterworth_hp(shape, cutoff=30, order=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-ccol, cols - ccol, cols)
    y = np.linspace(-crow, rows - crow, rows)
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt(xx**2 + yy**2)
    return 1 / (1 + (cutoff / (d + 1e-6)) ** order)


img = cv2.imread(DIR / "assets/image/descartes.jpg")
if img is None:
    raise ValueError("图像读取失败")

channels = cv2.split(img)
filter = butterworth_hp(channels[0].shape, cutoff=5, order=2)

processed_channels = [filter_channel(channel, filter) for channel in channels]
result = cv2.merge(processed_channels)

# 显示结果
cv2.imshow("Original", img)
cv2.imshow("Sharpened", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
