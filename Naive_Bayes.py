#importing the needed libraries
import numpy as np
import math
import pandas as pd

#get priors 
def get_prior(prior):
    c1 = 0
    c2 = 0
    for i in prior:
        if i == 0:
            c1+=1
        elif i == 1:
            c2+=1

    return c1/(c1+c2), c2/(c1+c2)

#get posteriors
def get_posterior(tweets, unique_features, y_train):
        c1 = 0
        c2 = 0
        for i in y_train:
            if i == 0:
                c1+=1
            elif i == 1:
                c2+=1
        p = np.full((2, len(unique_features)), 0.0)
        for i in range(0,len(unique_features)):
            feature_with_yes = 0
            feature_with_no = 0
            for j in range(0,len(tweets)):
                #check the class for each tweet and increment the counter for each word in features with it's corresponding tweet class
                for word in tweets[j]:
                    if word == unique_features[i]:
                        if y_train[j] == 0:
                            feature_with_no+=1
                        else:
                            feature_with_yes+=1
            #calculate the probability for each feature with yes/no class and add the smoothing
            p[0][i] = (feature_with_no + 0.01)/(c1 + (len(unique_features)* 0.01))
            p[1][i] = (feature_with_yes + 0.01)/(c2 + (len(unique_features)* 0.01))
        return p

#the Naive bayes method
def ML_NB(tweets_test, unique_features, p, p_c0, p_c1):
    y_predict = list()
    score = list()
    for i in range(0, len(tweets_test)):
        #calculate the log base 10 for the priors
        score_c0 = math.log10(p_c0)
        score_c1 = math.log10(p_c1)
        for word in tweets_test[i]:
            for j in range(0, len(unique_features)):
                if word == unique_features[j]:
                    #calculate the log base 10 for each word in tweet with yes/no class
                    score_c0 += math.log10(p[0][j])
                    score_c1 += math.log10(p[1][j])
        #compare the scores for each word with yes/no class and set y_predict value to the higher one's class
        if score_c0 > score_c1:
            y_predict.append("no")
            #write the score in scientific notation
            scientific_notation = "{:E}".format(score_c0)
            score.append(scientific_notation)
        else:
            y_predict.append("yes")
            #write the score in scientific notation
            scientific_notation = "{:E}".format(score_c1)
            score.append(scientific_notation)

    return y_predict, score

