import argparse
import logging
from pathlib import Path

import cv2
import imageio
import numpy as np
from skimage import img_as_float64, img_as_ubyte
from skimage.io import imread
from tqdm import tqdm


def gamma(image, gamma=2.2):
    image = np.power(img_as_float64(image), gamma)
    return img_as_ubyte(image)


def hdr(images, exposure, curve, output=None):
    images = [(imread(str(image_path))) for image_path in Path(images).rglob("*")]
    exposure = np.genfromtxt(exposure, delimiter=",")
    curve = np.exp(np.genfromtxt(curve, delimiter=","))

    hdr_image = np.zeros_like(images[0], dtype=np.float64)

    for k, image in tqdm(enumerate(images), total=len(images)):
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                for c in range(image.shape[2]):
                    # X_k(i,j) = E(i,j) * T_k
                    # E(i,j) = X_k(i,j) / T_k
                    hdr_image[i, j, c] += curve[image[i, j, c]][c] / exposure[k]

    hdr_image = hdr_image / len(images)
    hdr_rescaled = (hdr_image - hdr_image.min()) / (hdr_image.max() - hdr_image.min())

    if output:
        cv2.imwrite(output, cv2.cvtColor(img_as_ubyte(hdr_rescaled), cv2.COLOR_BGR2RGB))

    return hdr_image


def tone_map(image, alpha=0.18, output=None):
    L = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    N = image.shape[0] * image.shape[1]

    L_tilde = np.exp(np.log(L + 1e-8) / N)
    L_s = (alpha / L_tilde) * L
    L_g = L_s / (1 + L_s)

    tone_mapped = np.zeros_like(image)
    tone_mapped = L_g[:, :, None] * (image / L[:, :, None])
    tone_mapped = (tone_mapped - tone_mapped.min()) / (tone_mapped.max() - tone_mapped.min())

    if output:
        imageio.imsave(output, gamma(tone_mapped, 1./2.2))

    return tone_mapped


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description="Produces a HDR image from LDR images.")

    parser.add_argument(
        "--images",
        help="Path to a directory containing the LDR images to be processed.",
        required=True,
        type=str)

    parser.add_argument(
        "--exposure",
        help="Path to a file containing the exposure times for each image in `--image`.",
        required=True,
        type=str)

    parser.add_argument(
        "--curve",
        help="Path to a file containing camera response curve.",
        required=True,
        type=str)

    parser.add_argument(
        "--hdr-output",
        help="Name of the HDR file to be saved.",
        default="hdr_image.hdr",
        type=str)

    parser.add_argument(
        "--tone-mapped-output",
        help="Name of the tone mapped file to be saved.",
        default="tone_mapped.png",
        type=str)

    args = parser.parse_args()
    hdr_image = hdr(images=args.images, exposure=args.exposure, curve=args.curve, output=args.hdr_output)
    tone_map(hdr_image, output=args.tone_mapped_output)


if __name__ == "__main__":
    main()
