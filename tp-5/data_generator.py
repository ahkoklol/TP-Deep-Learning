import tensorflow as tf
import numpy as np


class SegmentationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, annotation_paths, batch_size, depth, shuffle=True):
        self.image_paths = image_paths
        self.annotation_paths = annotation_paths
        self.batch_size = batch_size
        self.depth = depth
        self.shuffle = shuffle
        self.on_epoch_end()
        self.cache = {}
        self.cache_tr = {}

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_annotation_paths = self.annotation_paths[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]

        images, annotations = tuple(
            map(
                list,
                zip(
                    *[
                        self.__get_datum_pair_from_cache(image_path, anno_path)
                        for image_path, anno_path in zip(
                            batch_image_paths, batch_annotation_paths
                        )
                    ]
                ),
            )
        )

        min_width = min(image.shape[1] for image in images)
        min_height = min(image.shape[0] for image in images)

        # Ensure dimensions are multiples of 2^depth
        min_width = min_width - (min_width % (2 ** (self.depth + 1)))
        min_height = min_height - (min_height % (2 ** (self.depth + 1)))

        processed_images = [
            self.crop_center(image, min_width, min_height) for image in images
        ]
        processed_annotations = [
            self.crop_center(anno, min_width, min_height) for anno in annotations
        ]

        return np.array(processed_images), np.array(processed_annotations)

    def __get_datum_pair_from_cache(self, image_path, anno_path):
        result = self.cache.get(image_path, None)

        if result is not None:
            return result, self.cache[anno_path]
        # else
        self.cache.update(
            {
                image_path: tf.keras.preprocessing.image.img_to_array(
                    tf.keras.preprocessing.image.load_img(image_path)
                )
                / 255.0,
                anno_path: tf.keras.preprocessing.image.img_to_array(
                    tf.keras.preprocessing.image.load_img(
                        anno_path, color_mode="grayscale"
                    )
                )
                / 255.0,
            }
        )

        return self.cache[image_path], self.cache[anno_path]

    def on_epoch_end(self):
        if self.shuffle:
            temp = list(zip(self.image_paths, self.annotation_paths))
            np.random.shuffle(temp)
            self.image_paths, self.annotation_paths = zip(*temp)

    def crop_center(self, img, crop_width, crop_height):
        mid_x, mid_y = img.shape[1] // 2, img.shape[0] // 2
        half_crop_width, half_crop_height = crop_width // 2, crop_height // 2
        return img[
            mid_y - half_crop_height : mid_y + half_crop_height,
            mid_x - half_crop_width : mid_x + half_crop_width,
        ]
