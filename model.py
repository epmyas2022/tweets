import helpers.data as data
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import helpers.utils as tools
import enums.tags as tags

tweets = data.DataReader("./data/sentiment-emotion-labelled_Dell_tweets.csv")
max_data = 25000

labels = tags.EnumModel().labels

num_classes = len(labels)
text_tweets = tweets.to_list("Text")[:max_data]
categories = tweets.replace_values(labels, "sentiment")[:max_data]

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")

preprocessed_text = tools.Utils.bert_preprocess(text_input)
outputs = tools.Utils.encoder(preprocessed_text)

layersl = tf.keras.layers.Dropout(0.1, name="dropout")(outputs["pooled_output"])


l = tf.keras.layers.Dense(num_classes, activation="softmax", name="output")(layersl)


model = tf.keras.Model([text_input], [l])

model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

model.fit(text_tweets, categories, epochs=10)


# Save model
model.save("model/mymodel")

# save weights

model.save_weights("model/mymodel/weights")
