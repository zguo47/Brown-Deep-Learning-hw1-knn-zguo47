import numpy as np
import tensorflow as tf
import tensorflow.keras.applications.resnet50 as resnet
from tqdm.auto import tqdm


class ResNetWrapper:
    #
    # DO NOT EDIT
    # ... but if your machine can't run the code,
    #     because it runs out of memeory,
    #     you can make the batch_size smaller in get_resnet_embeddings
    #
    # Don't worry if you don't understand everything in this class line by line.
    # You will learn about these things later in the course.
    def __init__(self):
        """
        Initialize the ResNet Wrapper here.
        NOTE: DO NOT EDIT
        """
        self.pretrained_model = resnet.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
            pooling=None,
        )

        self.pretrained_full_model = resnet.ResNet50(
            input_shape=(224, 224, 3), include_top=True, weights="imagenet"
        )

    def preprocess_image(self, image_uint):
        """
        Preprocess the images

        NOTE: DO NOT EDIT

        :param image_uint: raw images in uint8,
                           aranged in the (batch, height, width, channel) format
        :return image_preprocess: preprocessed images
        """
        image_resize = tf.image.resize(image_uint, [224, 224])
        image_preprocess = resnet.preprocess_input(image_resize)

        return image_preprocess

    def get_resnet_embeddings(self, image_preprocess):
        """
        Extract the ResNet embeddings from the preprocessed images

        NOTE: DO NOT EDIT ... but if your machine can't run the code,
        because it runs out of memeory,
        you can make the batch_size smaller.
        The default value is 10 here, but maybe you can change it to 5?

        Larger batch size means larger chunk of data is processed
            in parallel by vectorization, so the overall code executes faster.
            However, that also means larger memory space is needed.
            It's a typical case of CPU time vs. memory space compromise.

        :param image_preprocess: preprocessed images
        :return image_embedding: ResNet50 image embeddings
        """
        batch_size = 10
        train_size = len(image_preprocess)

        image_embedding = np.zeros((train_size, 7 * 7 * 2048), dtype=np.float32)

        for i in tqdm(range(0, train_size, batch_size)):
            image_embedding[i : i + batch_size] = (
                self.pretrained_model(image_preprocess[i : i + batch_size])
                .numpy()
                .reshape(batch_size, -1)
            )

        return image_embedding

    def get_full_model_predictions(self, image_preprocessed):
        """
        Return the predictions from the full ResNet model

        NOTE: DO NOT EDIT

        :param image_preprocess: preprocessed images
        :return preds_decoded: Decoded ResNet50 image predictions
        """
        preds = self.pretrained_full_model(image_preprocessed[np.newaxis, :]).numpy()
        preds_decoded = resnet.decode_predictions(preds, top=5)

        return preds_decoded
