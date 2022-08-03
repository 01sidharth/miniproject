
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding

df = pd.read_csv("./covid.csv")



tweet_df = df[['text','sentiment']]


print(tweet_df["sentiment"].value_counts())


sentiment_label = tweet_df.sentiment.factorize()



tweet = tweet_df.text.values
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweet)
vocab_size = len(tokenizer.word_index) + 1
encoded_docs = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(encoded_docs, maxlen=200)





embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(vocab_size, embedding_vector_length, input_length=200) )
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])  



model.fit(padded_sequence,sentiment_label[0],validation_split=0.2, epochs=5, batch_size=32)


print(tweet_df["sentiment"].value_counts())


from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def student():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form['Name']
      tw = tokenizer.texts_to_sequences([result])
      tw = pad_sequences(tw,maxlen=200)
      prediction = int(model.predict(tw).round().item())
      data=sentiment_label[1][prediction]
      return render_template("result.html",result1 = data)

if __name__ == '__main__':
   app.run(debug = True)
