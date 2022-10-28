# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 20:20:42 2022

@author: divyanshu
"""
import nltk

paragraph =  """Two-wheelers are the most preferred medium of transport in today’s world
 due to their compact size, efficiency, low maintenance, mobility for the Indian family, 
 and sheer joy while riding them. In the world India ranks 2nd position in production of 2 wheelers. 
 Today cars account for only 13 % of the vehicle population in India and on the other hand two-wheelers account for 70% of the vehicle population in India.
 Although two-wheelers are the most convenient medium of transport, they are the reason for serious traffic problems. According to the latest data from the NCRB 
 (National Crime Record Bureau) [Ministry of Home Affairs 2020], the rate of accidental death by two-wheelers increased by 43.6% in 2020 compared to 38% in 2019. 
 Though being the covid time there was 6 % rise in the rate of accidents by two-wheelers. This indicates that in covid time how people shifted to affordable and 
 safe transport mediums. India ranks 1st amongst 199 countries in road accident deaths reported in the World Road Statistics, 2018. According to the report by WHO on Road Safety 2018,11% of the accident-related deaths in the World are reported from India. According to the report by the world health organization (WHO), about 1.3 million people die every year worldwide as a result of accidents and 30 to 40 million people suffer injuries that result in lifelong disability [Ministry of Road Transport and Highway, 2022]."""
 
               
# Cleaning the texts
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet=WordNetLemmatizer()
sentences = nltk.sent_tokenize(paragraph)
corpus = []
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ', sentences[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the TF-IDF model
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
X = cv.fit_transform(corpus).toarray()