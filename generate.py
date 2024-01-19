import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import scipy as sp
from scipy import sparse
import re
import nltk
from sklearn.datasets import load_files
import os.path
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer


##############################################################################
#TOKENIZE TRAINING SET

# Import stopwords with nltk.
stop = stopwords.words('english')
new_stopwords = ['food', 'mcdonalds', 'burger', 'fries', 'one', 'all', 'like', 'just', 'about', 'some', 'very', 'more']
final_stop = stop + new_stopwords 
# Use English stemmer.
stemmer = SnowballStemmer("english")

#check if tokens are already generated
if not os.path.isfile('tokens.csv'):

    header_list = ["sentiment", "review"]
    df = pd.read_csv('train_file.csv', sep = ',', names=header_list)

    #remove html tags
    df["review"] = df['review'].str.replace('<[^<]+?>','')
    #remove punctuations
    df["review"] = df['review'].str.replace(r'[^\w\s]','')
    #remove digits
    df["review"] = df['review'].str.replace(r'[0-9]','')
    #convert to lowercase
    df["review"] = df['review'].str.lower()
    #remove stop words
    df["review"] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (final_stop)]))
    #split words
    df['review'] = df['review'].str.split()
    #stem words
    df['review'] = df['review'].apply(lambda x: [stemmer.stem(y) for y in x]) 
    #save the tokens as tokens.cvs
    df.to_csv('tokens.csv', index=False)

#read tokens again into pandas dataframe
df = pd.read_csv('tokens.csv')
print("1. TRAINING SET TOKENIZED")
##############################################################################
#VECTORIZE TRAINING SET

v = TfidfVectorizer()
train_matrix = v.fit_transform(df['review'])

#convert to csr matrix to index
train_matrix = train_matrix.tocsr()


sentiment_col = df[["sentiment"]].to_numpy()
print("2. TRAINING SET VECTORIZED")
##############################################################################
#TOKENIZE TESTING SET

if not os.path.isfile('tokens_test.csv'):

    a_file = open("test_file.txt", "r")

    test_lines = []
    for line in a_file:
      linestripped = line.strip()
      test_lines.append(linestripped)

    a_file.close()


    df = pd.DataFrame(test_lines, columns = ['review'])

    #remove html tags
    df["review"] = df['review'].str.replace('<[^<]+?>','')
    #remove punctuations
    df["review"] = df['review'].str.replace(r'[^\w\s]','')
    #remove digits
    df["review"] = df['review'].str.replace(r'[0-9]','')
    #convert to lowercase
    df["review"] = df['review'].str.lower()
    #remove stop words
    df["review"] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (final_stop)]))
    #split words
    df['review'] = df['review'].str.split()
    #stem words
    df['review'] = df['review'].apply(lambda x: [stemmer.stem(y) for y in x]) 

    #save the tokens as tokens.cvs
    df.to_csv('tokens_test.csv', index=False)

dftest = pd.read_csv('tokens_test.csv')
print("3. TESTING SET TOKENIZED")

##############################################################################
#VECTORIZE TESTING SET

#vectorize the review columns
test_matrix = v.transform(dftest['review'])

#convert to csr matrix to index
test_matrix = test_matrix.tocsr()

print("4. TESTING SET VECTORIZED")

##############################################################################

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=100)
neigh.fit(train_matrix, sentiment_col)


result = []
for i in range(test_matrix.shape[0]):
    prediction = neigh.predict(test_matrix[i,:])
    print("i: %s prediction: %s" % (i, prediction))

    result.append(prediction)

#train_matrix.shape[0]
result = [i[0] for i in result]

textfile = open("generated.txt", "w")
for element in result:
    textfile.write(str(element))
    textfile.write('\n')
textfile.close()


