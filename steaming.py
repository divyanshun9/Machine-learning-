import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph = """Two-wheelers are the most preferred medium of transport in today’s world
 due to their compact size, efficiency, low maintenance, mobility for the Indian family, 
 and sheer joy while riding them. In the world India ranks 2nd position in production of 2 wheelers. 
 Today cars account for only 13 % of the vehicle population in India and on the other hand two-wheelers account for 70% of the vehicle population in India.
 Although two-wheelers are the most convenient medium of transport, they are the reason for serious traffic problems. According to the latest data from the NCRB 
 (National Crime Record Bureau) [Ministry of Home Affairs 2020], the rate of accidental death by two-wheelers increased by 43.6% in 2020 compared to 38% in 2019. 
 Though being the covid time there was 6 % rise in the rate of accidents by two-wheelers. This indicates that in covid time how people shifted to affordable and 
 safe transport mediums. India ranks 1st amongst 199 countries in road accident deaths reported in the World Road Statistics, 2018. According to the report by WHO on Road Safety 2018,11% of the accident-related deaths in the World are reported from India. According to the report by the world health organization (WHO), about 1.3 million people die every year worldwide as a result of accidents and 30 to 40 million people suffer injuries that result in lifelong disability [Ministry of Road Transport and Highway, 2022]."""
 
sentences = nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

# Stemming
for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)   
    
    
    
    
    
    
    
    
    