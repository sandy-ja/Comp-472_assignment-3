#importing the needed libraries
import pandas as pd
import re
import nltk
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from pandas import options

#extracting data from the training file
def extract_data(file_name):
    options.display.max_colwidth = 1000
    #read the training file as table
    data = pd.read_table(file_name)
    data_text=data['text'].unique()
    y_label = data['q1_label']
    data_old = data_text
    tweets = list()
    words = set(nltk.corpus.words.words())

    for i in data_old:
        #take only the alphabet characters
        i = re.sub('[^A-Za-z]', ' ', i)
        #take only english words that their length is more than 2 
        x = " ".join(w for w in nltk.wordpunct_tokenize(i) if w.lower() in words and len(w)>2)
        #convert the words to lower case
        x = x.lower()
        #tokenize the tweet
        k = word_tokenize(x)
        #append the tokenized tweet to the tweets list
        tweets.append(k)

    x =  np.array2string(data_text, precision=2, separator=',',suppress_small=True)
    #take only the alphabet characters
    x = re.sub('[^A-Za-z]', ' ', x)
    words = set(nltk.corpus.words.words())
    x = " ".join(w for w in nltk.wordpunct_tokenize(x) if w.lower() in words and len(w)>2)
    x = x.lower()
    features = word_tokenize(x)

    #remove stop words
    for word in features:
        if word in stopwords.words('english'):
            features.remove(word)
            
    words = dict()
    for word in features:
        words[word] = words.get(word,0) + 1

    #initialize liss for unique and filtered features
    unique_features = list()
    filtered_features = list()
    for k,v in words.items(): # a loop with two iteration Variables
        #append the words to unique_features
        unique_features.append(k)
        if v>1:
            #append only the words that repeated more than once to filtered_features
            filtered_features.append(k)

    #initialize x filtered train and x train matrices to zeros
    x_train_filtered = np.full((len(tweets), len(filtered_features)), 0)
    x_train = np.full((len(tweets), len(unique_features)), 0)

    #fill the filtered x train and x train 
    for i in tweets:
        for j in range(0 ,len(i)):
            for k in unique_features:
                if i[j] == k:
                    x_train [tweets.index(i)][unique_features.index(k)]+=1
            for k in filtered_features:
                if i[j] == k:
                    x_train_filtered [tweets.index(i)][filtered_features.index(k)]+=1

    #replacing yes and no in q1_label to 1's and 0's
    y = pd.Series(np.where(y_label.values == 'yes', 1, 0),y_label.index)
    y_train = y.to_numpy()

    return x_train_filtered, x_train, y_train, tweets, unique_features, filtered_features

#extracting data from the test file
def extract_test_data(fileName):
    #read the file as a table
    data = pd.read_table(fileName, header=None)
    #split the id column
    id_test = data.iloc[:, 0]
    #split the text column
    data_text = data.iloc[:, 1]
    y_test = data.iloc[:, 2]
    tweets = list()
    words = set(nltk.corpus.words.words())

    for i in data_text:
        i = re.sub('[^A-Za-z]', ' ', i)
        x = " ".join(w for w in nltk.wordpunct_tokenize(i) if w.lower() in words and len(w)>2)
        x = x.lower()
        k = word_tokenize(x)
        tweets.append(k)

    return tweets, y_test, id_test


#calculate the accuracy 
def Accuracy(y_test, y_predict):
    acc = 0
    accuracy = 0
    #if y_predict equals y_test increment the counter
    for i in range(0, len(y_predict)): 
        if y_predict[i] == y_test[i]:
            acc+=1
    #calculate accuracy by deviding the counter on the y_predict length
    accuracy = acc/len(y_predict)

    return accuracy

#a function to calculate Precision, Recall, and F1
def Precision_Recall_F1(y_test, y_predict):
    tp_yes, fp_yes, fn_yes, tp_no, fp_no, fn_no, per_yes, per_no, rec_yes, rec_no, f_yes, f_no = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    for i in range(0, len(y_predict)): 
        if y_predict[i] == 'yes' and y_test[i] == 'yes':
            tp_yes+=1
        if y_predict[i] == 'yes' and y_test[i] == 'no':
            fp_yes+=1
            fn_no+=1
        if y_predict[i] == 'no' and y_test[i] == 'no':
            tp_no+=1
        if y_predict[i] == 'no' and y_test[i] == 'yes':
            fp_no+=1
            fn_yes+=1

    #calculate precision, recall, and f1 for both classes yes and no
    per_yes = (tp_yes/(tp_yes+fp_yes))
    per_no = (tp_no/(tp_no+fp_no))
    rec_yes = (tp_yes/(tp_yes+fn_yes))
    rec_no = (tp_no/(tp_no+fn_no))
    f_yes = (2*per_yes*rec_yes)/(per_yes+rec_yes)
    f_no = (2*per_no*rec_no)/(per_no+rec_no)

    return per_yes, per_no, rec_yes, rec_no, f_yes, f_no


       

