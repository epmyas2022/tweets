
import tensorflow as tf
import tensorflow_hub as hub

BERT_PREPROCESSING = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
BERT_MODEL = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1'

class Utils:
    @staticmethod
    def encoder(layer: hub.KerasLayer, url_bert_model: str = BERT_MODEL) -> hub.KerasLayer:
        bert_encode = hub.KerasLayer(url_bert_model)
        return bert_encode(layer)
    
    @staticmethod
    def text_input(name='text') -> tf.keras.layers.Input:
        return tf.keras.layers.Input(shape=(), dtype=tf.string, name=name)
    
    @staticmethod
    def bert_preprocess(inputKeras: tf.keras.layers.Input, url_bert_preprocessing: str = BERT_PREPROCESSING) -> hub.KerasLayer:
        return hub.KerasLayer(url_bert_preprocessing)(inputKeras)