import tensorflow as tf
from datasets import Dataset
import tensorflow_hub as hub
import tensorflow_text as text



#load model weights
model = tf.keras.models.load_model('./model/mymodel')


datatest = [
   'I hate having to go out on the weekend.',
   'I didnt play that well',
   'The forms to improve the secretary of innovation',
   'One of the best ideas is to invest in artificial intelligence',
   'If you dont study you will get a bad grade',
   'because nobody understands me',
  'I like you a lot'
]

labels = {
    'negative': 0,
    'positive': 1,
    'neutral': 2
}


predict = model.predict(datatest)
for i in range(len(predict)):
    print('Text: ', datatest[i])
    #print('Real: ', output[i] == 4 and 'positive' or 'negative')
    print('Predict: ', list(labels.keys())[predict[i].argmax()])
    print('-----------------------') 