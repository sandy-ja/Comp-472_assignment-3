#importing the needed libraries and classes
import pandas as pd
import numpy as np
import multi_functional
import Naive_Bayes
import Output
import csv

#calling extract_data function 
x_train_filtered, x_train, y_train, tweets, unique_features, filtered_features = multi_functional.extract_data('covid_training.tsv')

#get priors
p_c0, p_c1 = Naive_Bayes.get_prior(y_train)

#get posteriors
p = Naive_Bayes.get_posterior(tweets, unique_features, y_train)
p_f = Naive_Bayes.get_posterior(tweets, filtered_features, y_train)

#calling extract_test_data
tweets_test , y_test, id_test= multi_functional.extract_test_data('covid_test_public.tsv')

#calling the Naive_Bayes model
y_predict, score = Naive_Bayes.ML_NB(tweets_test, unique_features, p, p_c0, p_c1)
y_predict_f, score_f = Naive_Bayes.ML_NB(tweets_test, filtered_features, p_f, p_c0, p_c1)

#calculate the accuracy
accuracy = multi_functional.Accuracy(y_test, y_predict)
accuracy_f = multi_functional.Accuracy(y_test, y_predict_f)

#calculate precision, recall, f1
per_yes, per_no, rec_yes, rec_no, f_yes, f_no= multi_functional.Precision_Recall_F1(y_test, y_predict)
per_yes_f, per_no_f, rec_yes_f, rec_no_f, f_yes_f, f_no_f= multi_functional.Precision_Recall_F1(y_test, y_predict_f)

#creating trace and evaluation files for OV data
Output.Trace_file("OV", id_test, y_predict, y_test, score)
Output.Eval_file("OV", accuracy, per_yes, per_no, rec_yes, rec_no, f_yes, f_no)

#creating trace and evaluation files for FV data
Output.Trace_file("FV", id_test, y_predict_f, y_test, score_f)
Output.Eval_file("FV", accuracy_f, per_yes_f, per_no_f, rec_yes_f, rec_no_f, f_yes_f, f_no_f)


