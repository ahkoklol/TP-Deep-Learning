import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tp5_utils import ResidualConv2D, UNetEncoder, UNetDecoder
import argparse


def crop_center(img, crop_width, crop_height):
    mid_x, mid_y = img.shape[1] // 2, img.shape[0] // 2
    half_crop_width, half_crop_height = crop_width // 2, crop_height // 2
    return img[
        mid_y - half_crop_height : mid_y + half_crop_height,
        mid_x - half_crop_width : mid_x + half_crop_width,
    ]


def prepare_data(data_path, DEPTH):
    image = Image.open(data_path)

    r = max(image.size[0] // 1000, image.size[1] // 1000, 1)

    image = image.resize((image.size[0] // r, image.size[1] // r))

    image = np.asarray(image) / 255.0

    min_width = image.shape[1] - (image.shape[1] % (2 ** (DEPTH + 1)))
    min_height = image.shape[0] - (image.shape[0] % (2 ** (DEPTH + 1)))

    image = crop_center(image, min_width, min_height)

    return image


def predict(data_path, model_path, save_path, DEPTH):
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "ResidualConv2D": ResidualConv2D,
            "UNetEncoder": UNetEncoder,
            "UNetDecoder": UNetDecoder,
        },
    )

    image = prepare_data(data_path, DEPTH)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(model.predict(image[np.newaxis, :])[0, ...])
    axes[1].set_title("Nails Prediction")
    axes[1].axis("off")

    # Save the figure to a file
    plt.savefig(save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Image processing script.")
    parser.add_argument("-i", "--input", required=True, help="Input image file")
    parser.add_argument("-m", "--model", required=True, help="AI model to use")
    parser.add_argument("-o", "--output", required=True, help="Output image file name")
    parser.add_argument(
        "-d", "--depth", type=int, required=True, help="Depth parameter"
    )

    args = parser.parse_args()

    predict(args.input, args.model, args.output, args.depth)


if __name__ == "__main__":
    main()
