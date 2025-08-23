import random

from janim.imports import *
import numpy as np
import cv2
from cv2.typing import MatLike
from PIL.Image import Image

with reloads():
    from common import *


# 裁剪图片为正方形
def imageCrop(img: MatLike) -> MatLike:
    height, width = img.shape[:2]
    if width > height:
        start_width = (width - height) // 2
        end_width = start_width + height
        return img[:, start_width:end_width, :]
    else:
        start_height = (height - width) // 2
        end_height = start_height + width
        return img[start_height:end_height, :, :]


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


def relDistances(shape):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    x = np.linspace(-ccol, cols - ccol, cols)
    y = np.linspace(-crow, rows - crow, rows)
    xx, yy = np.meshgrid(x, y)
    d = np.sqrt((xx / ccol) ** 2 + (yy / crow) ** 2)
    return d


def blur(shape, radius=0.1):
    d = relDistances(shape)
    filter = np.ones_like(d)
    filter[d > radius] = 0
    return filter


def get_fft_image(image: MatLike, filter: MatLike | None = None) -> MatLike:
    channels = cv2.split(image)
    channels_fft = (fdMap(ch) for ch in channels)
    if filter is not None:
        channels_fft = (filter_channel(ch, filter) for ch in channels_fft)
    return cv2.merge(tuple(channels_fft))


def toPILImage(cv2_image: MatLike) -> Image:
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


class ImageGrid(Group[Square]):
    def __init__(
        self, img: MatLike, width: float | None = 5, height: float | None = None, **args
    ):
        super().__init__(**args)
        img_height, img_width = img.shape[:2]
        if height is None:
            if width is None:
                raise ValueError("Either width or height must be specified")
            else:
                height = width * img_height / img_width
        elif width is None:
            width = height * img_width / img_height

        for i, img_row in enumerate(img):
            for j, img_pixel in enumerate(img_row):
                b, g, r = img_pixel
                color = f"#{r:02x}{g:02x}{b:02x}"
                x0 = j / img_width * width
                y0 = i / img_height * height
                x1 = (j + 1) / img_width * width
                y1 = (i + 1) / img_height * height
                i_pixel = Rect(
                    (x0, -y0, 0),
                    (x1, -y1, 0),
                    stroke_radius=0.005,
                    stroke_color=WHITE,
                    fill_alpha=1,
                    fill_color=color,
                )
                self.add(i_pixel)


def fftComplex(ch):
    result = cv2.dft(np.float32(ch), flags=cv2.DFT_COMPLEX_OUTPUT)
    return result[:, :, 0] + 1j * result[:, :, 1]


def splitComplex(ch_complex):
    result = np.empty((*ch_complex.shape, 2), dtype=ch_complex.real.dtype)
    result[:, :, 0] = ch_complex.real
    result[:, :, 1] = ch_complex.imag
    return result


def imageFFT(img: MatLike) -> MatLike:
    return np.transpose(
        [np.fft.fftshift(fftComplex(ch)) for ch in cv2.split(img)], (1, 2, 0)
    )


def imageIFFT(img_fft: MatLike) -> MatLike:
    return cv2.merge(
        tuple(
            cv2.normalize(
                cv2.idft(
                    splitComplex(np.fft.ifftshift(ch)),
                    flags=cv2.DFT_REAL_OUTPUT,
                ),
                None,
                0,
                255,
                cv2.NORM_MINMAX,
            ).astype(np.uint8)
            for ch in np.transpose(img_fft, (2, 0, 1))
        )
    )


def fdMap(img_fft: np.ndarray) -> MatLike:
    mags = 20 * np.log(np.abs(img_fft) + 1e-6)
    return cv2.normalize(mags, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


class TL_ImageFilter(Timeline):
    CONFIG = config

    def construct(self):
        imageFile = DIR / "assets/image/cxk.png"
        img_size = (32, 32)

        img = cv2.imread(imageFile)
        assert img is not None, f"Failed to load image {imageFile}"
        print(img.shape)
        img = cv2.resize(imageCrop(img), img_size)

        # 傅里叶变换
        img_fft = imageFFT(img)
        img_fdmap = fdMap(img_fft)

        brightDotPos = (0, 0)
        max_amp = np.max(np.abs(img_fft))
        img_processed_fft = img_fft.copy()
        img_processed_fft[brightDotPos] *= max_amp / np.abs(
            img_processed_fft[brightDotPos]
        )

        img_processed_fdmap = fdMap(img_processed_fft)
        img_processed = imageIFFT(img_processed_fft)

        i_img = ImageGrid(img, width=5).points.to_center().r
        i_img_fdmap = ImageGrid(img_fdmap, width=5).points.to_center().r
        Group(i_img, i_img_fdmap).points.arrange(RIGHT, buff=1).to_center().shift(
            UP * 0.5
        )
        i_img_processed_fdmap = (
            ImageGrid(img_processed_fdmap, width=5).points.move_to(i_img_fdmap).r
        )
        i_img_processed = ImageGrid(img_processed, width=5).points.move_to(i_img).r

        random.seed(42)
        ag = [FadeIn(item) for item in i_img]
        random.shuffle(ag)
        self.play(*ag, lag_ratio=0.01, duration=1)
        self.forward(1)
        ag = [
            Transform(
                item1,
                item2,
                hide_src=False,
                path_arc=interpolate(-PI / 2, PI / 2, random.random()),
            )
            for item1, item2 in zip(i_img, i_img_fdmap)
        ]
        random.shuffle(ag)
        self.play(*ag, lag_ratio=1e-2, duration=5)
        self.forward(2)
        self.play(
            FadeIn(i_img_processed_fdmap),
            FadeIn(i_img_processed),
            duration=3,
        )

        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
#########################################################
