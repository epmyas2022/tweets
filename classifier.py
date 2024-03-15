import tensorflow as tf
import tensorflow_text as text

import enums.tags as tags
class Classifier:
    labels = tags.EnumModel().labels

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def classify(self, data):
        return self.model.predict(data)
