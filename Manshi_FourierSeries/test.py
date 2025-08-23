from pathlib import Path

import cv2
import numpy as np

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()


def process_channel(channel):
    float_ch = np.float32(channel)
    dft = cv2.dft(float_ch, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(
        cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-6
    )
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


img = cv2.imread(DIR / "assets/image/cxk.png")
if img is None:
    raise ValueError("图像读取失败")

b, g, r = cv2.split(img)
spectrum_b = process_channel(b)
spectrum_g = process_channel(g)
spectrum_r = process_channel(r)

spectrum_merged = cv2.merge([spectrum_b, spectrum_g, spectrum_r])

cv2.imshow("Original", img)
cv2.imshow("Color Spectrum", spectrum_merged)
cv2.waitKey(0)
