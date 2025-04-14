# %% Import libraries
import pandas as pd
import numpy as np

review = pd.read_excel('Dataset/Data ulasan Shopee tentang COD(2).xlsx')
print(review.head(5))

# %% Tokenisasi
import pandas as pd
import string
import re
import json
from nlp_id.tokenizer import Tokenizer
from nlp_id.stopword import StopWord 
from nlp_id.lemmatizer import Lemmatizer 

#import kamus bahasa baku
with open('Dataset/combined_slang_words.txt') as f:
    data0 = f.read()
print("Data type before reconstruction : ", type(data0))
formal_indo = json.loads(data0)
print("Data type after reconstruction : ", type(formal_indo), '\n')

def informal_to_formal_indo(text):
    res = " ".join(formal_indo.get(ele, ele) for ele in text.split())
    return(res)

tokenizer = Tokenizer()
stopword = StopWord()
lemmatizer = Lemmatizer()

def my_tokenizer(doc):
    doc = re.sub(r'@[A-Za-z0-9]+', '', doc)
    doc = re.sub(r'#[A-Za-z0-9]+', '', doc)
    doc = re.sub(r'RT[\s]', '', doc)
    doc = re.sub(r"http\S+", '', doc)
    doc = re.sub(r'[0-9]+', '', doc)
    doc = re.sub(r'(.)\1+',r'\1\1', doc)
    doc = re.sub(r'[\?\.\!]+(?=[\?.\!])', '',doc)
    doc = re.sub(r'[^a-zA-Z]',' ', doc)
    doc = re.sub(r'\b(\w+)( \1\b)+', r'\1', doc)
    doc = doc.replace('\n', ' ')
    doc = doc.translate(str.maketrans('', '', string.punctuation))
    doc = doc.strip(' ')
    #Mengubah menjadi huruf kecil
    doc = doc.lower()
    #Text Normalization
    doc = informal_to_formal_indo(doc)
    #Punctuation Removal+Menghapus Angka
    doc = doc.translate(str.maketrans('', '', string.punctuation + string.digits))
    #Whitespace Removal
    doc = doc.strip()
    #Tokenization
    doc = tokenizer.tokenize(doc)
    doc_token1 = [word for word in doc]
    #Stopwords Removal
    doc_token2 = [word for word in doc_token1 if word not in stopword.get_stopword()]
    #Lemmatization
    doc_token3 = [lemmatizer.lemmatize(word) for word in doc_token2]
    return doc_token3

#  Apply text pre-processing
review['preprocessing'] = review['content'].apply(my_tokenizer)
print(review[['content', 'preprocessing']])

# %% Stemming
review1 = review[['content', 'preprocessing']]

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemming(ulasan) :
  do = []
  for w in ulasan:
    dt = stemmer.stem(w)
    do.append(dt)

  d_clean = []
  d_clean = " ".join(do)
  return d_clean

review['stemming_ulasan'] = review['preprocessing'].apply(stemming)
print('Stemming Result : \n')
print(review[['stemming_ulasan']])

# %% Sentiment Analysis with Kamus
lexicon_positive = pd.read_excel('Dataset/kamus_positif.xlsx')
lexicon_positive_dict = {}
for index, row in lexicon_positive.iterrows():
    if row[0] not in lexicon_positive_dict:
        lexicon_positive_dict[row[0]] = row[1]

lexicon_negative = pd.read_excel('Dataset/kamus_negatif.xlsx')
lexicon_negative_dict = {}
for index, row in lexicon_negative.iterrows():
    if row[0] not in lexicon_negative_dict:
        lexicon_negative_dict[row[0]] = row[1]

def sentiment_analysis_lexicon_indonesia(ulasan):
    score = 0
    for word in ulasan:
        if (word in lexicon_positive_dict):
            score = score + lexicon_positive_dict[word]
    for word in ulasan:
        if (word in lexicon_negative_dict):
            score = score + lexicon_negative_dict[word]
    sentimen=''
    if (score > 0):
        sentimen = 'positif'
    elif (score < 0):
        sentimen = 'negatif'
    else:
        sentimen = 'netral'
    return score, sentimen

results = review['preprocessing'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
review['label'] = results[0]

# Masih hasil Copy Paste
# data['sentimen'] = results[1] 
# data

review['label'] = results[1]
dataSentimen = review
data_inset = review

data_inset[['content', 'preprocessing', 'label']]

# %% Check label sentiment analysis
data = review[['stemming_ulasan', 'label']]
data

# %% Cek jumlah masing-masing label
data['label'].value_counts()

# %% Plotting label dengan persen dengan bentuk piechart
data['label'].value_counts().plot.pie(autopct='%.2f')

# %% Change value label jadi angka
data.replace(to_replace='negatif', value=0, inplace=True)
data.replace(to_replace='positif', value=1, inplace=True)
data.replace(to_replace='netral', value=2, inplace=True)
data.head(10)

# %% Split data training, validation and test
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(data, test_size=0.2)
df_val, df_test = train_test_split(df_test, test_size=0.5)
df_train.shape, df_test.shape, df_val.shape
print('Training data shape:', df_train.shape)
print('Validation data shape:', df_val.shape)
print('Test data shape:', df_test.shape)

# %% Membuat data training ke file csv
df_train.to_csv('data_training.csv', index = False)
data = pd.read_csv('data_training.csv')
data.head()

# %% Membuat data validasi ke file csv
df_val.to_csv('data_validasi.csv', index = False)
data = pd.read_csv('data_validasi.csv')
data.head()
# %% Membuat data test ke file csv
df_test.to_csv('data_testing.csv', index = False)
data = pd.read_csv('data_testing.csv')
data.head()

# %% Import BERT
from transformers import BertTokenizer

# Load tokenizer dari pre-trained model
bert_tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p2')

# %% View vocabulary from pre-trained models that have been preloaded
vocabulary = bert_tokenizer.get_vocab()
print('Panjang vocabulary:', len(vocabulary))

# %% Example of Tokenization
# Retrieve the 1st index data on the dataframe
print('Kalimat:', review['stemming_ulasan'][0])
print('BERT Tokenizer:', bert_tokenizer.tokenize(review['stemming_ulasan'][0]))

# %% Example of input formatting for BERT.
# Input formatting can use 'encode_plus' function
bert_input = bert_tokenizer.encode_plus(
    # Sample sentences
    review['stemming_ulasan'][0],
    # Add [CLS] token at the beginning   of the sentence & [SEP] token at the end of the sentence
    add_special_tokens = True,
    # Add padding to max_length using [PAD] token
    # jika kalimat kurang dari max_length
    padding = 'max_length',
    # Truncate if sentence is more than max_length
    truncation = 'longest_first',
    # Determine the max_length of the entire sentence
    max_length = 50,
    # Returns the attention mask value
    return_attention_mask = True,
    # Returns the value of token type id (segment embedding)
    return_token_type_ids =True)
# The function 'encode_plus' returns 3 values:
# input_ids, token_type_ids, attention_mask
bert_input.keys()

# %% Original data
print('Kalimat\t\t:', review['stemming_ulasan'][0]) #1 denotes first order data or first review data
# so for example I change it to 1000 still 1 data appears but the order is 1000th
# Input formatting + tokenizer return
print('Tokenizer\t:', bert_tokenizer.convert_ids_to_tokens(bert_input['input_ids']))

# Input IDs: token indexes in the tokenizer vocabulary
print('Input IDs\t:', bert_input['input_ids'])

# Token type IDs: shows the sequence of sentences in the sequence (segment embedding)
print('Token Type IDs\t:', bert_input['token_type_ids'])

# Attention mask : returns value [0,1].
#1 means masked token, 0 tokens are not masked (ignored)
print('Attention Mask\t:', bert_input['attention_mask'])

# %% 
import seaborn as sns
import matplotlib.pyplot as plt
# There are many ways to define max_length
# The intuition is that we don't want to cut sentences
# Or added too much padding (longer computation)

# In this example, max_length is determined from the distribution of tokens in the dataset
token_lens = []
for txt in review['stemming_ulasan']:
  tokens = bert_tokenizer.encode(txt)
  token_lens.append(len(tokens))
sns.histplot(token_lens, kde=True, stat='density', linewidth=0)
plt.xlim([0, 100])
plt.xlabel('Token count')

# %% Create a function to combine tokenization steps
# Added special tokens for all data as input formatting to the BERT model
def convert_example_to_feature(sentence):
  return bert_tokenizer.encode_plus(
      sentence,
      add_special_tokens=True,
      padding='max_length',
      truncation='longest_first',
      max_length=42,
      return_attention_mask=True,
      return_token_type_ids=True)

# %% Create a function to map input formatting results to match the BERT model
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
  return {
      "input_ids": input_ids,               # Sebagai token embedding
      "token_type_ids": token_type_ids,     # Sebagai segment embedding
      "attention_mask": attention_masks,    # Sebagai filter informasi mana yang kalkulasi oleh model
  }, label

# %% 
import tensorflow as tf
# Create a function to iterate or encode each sentence in the entire data
def encode(data):
  input_ids_list = []
  token_type_ids_list = []
  attention_mask_list = []
  label_list = []

  for sentence, label in data.to_numpy():
    bert_input = convert_example_to_feature(sentence)
    input_ids_list.append(bert_input['input_ids'])
    token_type_ids_list.append(bert_input['token_type_ids'])
    attention_mask_list.append(bert_input['attention_mask'])
    label_list.append([label])
  return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

# %% Perform input formatting using the previous function on the data as a whole
train_encoded = encode(df_train).batch(32)
test_encoded = encode(df_test).batch(32)
val_encoded = encode(df_val).batch(32)

# %% Download Model
from transformers import TFBertForSequenceClassification

# Load model
bert_model = TFBertForSequenceClassification.from_pretrained(
    'indobenchmark/indobert-base-p2', num_labels=3)

# %% Compile model
bert_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00003),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=tf.keras.metrics.SparseCategoricalAccuracy('accuracy'))

# %% Epoch
import time 

start_time = time.time()

bert_history = bert_model.fit(train_encoded, epochs=5,
                              batch_size=32, validation_data=val_encoded)

end_time = time.time()

print(f"Waktu Training : {end_time - start_time:.2f} detik")

# %% Create a function for plotting training results
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel('Epochs')
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()

plot_graphs(bert_history, 'accuracy')
plot_graphs(bert_history, 'loss')