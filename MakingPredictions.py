import os
from dataclasses import dataclass, field
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
#  pip install keras==2.4.3 tensorflow==2.3.1 numpy==1.18.5


@dataclass
class CaptchaSolver:
    model_path: str
    characters: tuple = field(
        default=('2', '3', '4', '5', '6', '7', '8', 'b', 'c', 'd', 'e', 'f', 'g', 'm', 'n', 'p', 'w', 'x', 'y'),
        init=False)
    batch_size: int = field(default=16, init=False)
    img_width: int = field(default=200, init=False)
    img_height: int = field(default=50, init=False)
    max_length: int = field(default=5, init=False)

    def __post_init__(self):
        self.char_to_num, self.num_to_char = self.get_string_lookups()
        self.prediction_model = self.load_tf_model()

    def load_tf_model(self) -> tf.keras.models.load_model:
        model = tf.keras.models.load_model(self.model_path)
        prediction_model = keras.models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense2").output
        )
        return prediction_model

    def get_string_lookups(self) -> List[layers.experimental.preprocessing.StringLookup]:

        char_to_num = layers.experimental.preprocessing.StringLookup(
            vocabulary=list(list(self.characters)), num_oov_indices=0, mask_token=None
        )

        num_to_char = layers.experimental.preprocessing.StringLookup(
            vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

        return [char_to_num, num_to_char]

    def encode_single_sample(self, img_path, label):
        img = tf.io.read_file(img_path)
        img = tf.io.decode_png(img, channels=1)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, [self.img_height, self.img_width])
        img = tf.transpose(img, perm=[1, 0, 2])
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

        return {"image": img, "label": label}

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
                  :, :self.max_length
                  ]
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text

    def make_pred(self, img_paths_list: list):
        validation_dataset = tf.data.Dataset.from_tensor_slices((img_paths_list, ['' for _ in range(len(img_paths_list))]))
        validation_dataset = (
            validation_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            ).batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )
        for batch in validation_dataset.take(1):
            batch_images = batch["image"]
            batch_labels = batch["label"]

            preds = self.prediction_model.predict(batch_images)  # reconstructed_model is saved trained model
            pred_texts = self.decode_batch_predictions(preds)

            #  plotting stuff
            # orig_texts = []
            # for label in batch_labels:
            #     label = tf.strings.reduce_join(self.num_to_char(label)).numpy().decode("utf-8")
            #     orig_texts.append(label)

            # _, ax = plt.subplots(2, 4, figsize=(15, 5))
            # for i in range(len(pred_texts)):
            #     img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
            #     img = img.T
            #     title = f"Prediction: {pred_texts[i]}"
            #
            #     ax[i//4, i % 4].imshow(img, cmap="gray")
            #     ax[i//4, i % 4].set_title(title)
            #     ax[i//4, i % 4].axis("off")
        # plt.show()
        return pred_texts


if __name__ == '__main__':
    images = [rf"Images\{img}" for img in os.listdir("Images")]
    print(images)
    captcha_solver = CaptchaSolver(r"model_tf")
    print("Model loaded")
    predictions = captcha_solver.make_pred(images)
    print(predictions)
