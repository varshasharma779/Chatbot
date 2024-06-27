import numpy as np
import nltk
import string
import random

f = open(r'C:\Users\varsh\Downloads\python\jcbose.txt', 'r', errors='ignore')
raw_doc = f.read()
raw_doc = raw_doc.lower()#converting entire text to lowercase
nltk.download('punkt')#using the punkt tokenizer
nltk.download('wordnet')#using the wordnet dictionary
nltk.download('omw-1.4')
sentence_tokens = nltk.sent_tokenize(raw_doc) #using sentence tokenizer in row doc
word_tokens = nltk.word_tokenize(raw_doc)# using word tokenizer in row doc


lemmer = nltk.stem.WordNetLemmatizer()  # perform lemetization and remove puncuations
def LemTokens(tokens):
  return [lemmer.lemmatize(token) for token in tokens]
remove_punc_dict = dict((ord(punct),None) for punct in string.punctuation)
def LemNormalize(text):
  return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punc_dict )))


greet_inputs = ('hey','hello','hi','hii','whassup','how are you')
greet_responses = ('whassup','hey','hey there!','hello','hi how are you?')
def greet(sentence):
  for word in sentence.split():
    if word.lower() in greet_inputs:
      return random.choice(greet_responses)

from sklearn.feature_extraction.text import TfidfVectorizer #to conver words in numeric form or numbers . machine learning (machine can not understand text)

from sklearn.metrics.pairwise import cosine_similarity # to check similarity between two texts


def response(user_response):
  robo1_response = ''
  TfidfVec =TfidfVectorizer(tokenizer = LemNormalize, stop_words = 'english') #  tokenize , lemetize, remove stop words
  tfidf = TfidfVec.fit_transform(sentence_tokens) # convert every word into tfidf vector
  vals = cosine_similarity(tfidf[-1],tfidf)# check the similar sentences
  idx = vals.argsort()[0][-2]# return the 0th means most similar responce or 2nd
  flat = vals.flatten()
  flat.sort() # sort the words
  req_tfidf = flat[-2]
  if (req_tfidf == 0):
    robo1_response = robo1_response + "I am sorry. unable to understand you!"
    return robo1_response
  else:
    robo1_response = robo1_response+ sentence_tokens[idx]
    return robo1_response

flag = True
print('hello! I am the learning bot.')
while(flag == True):
  user_response = input()
  user_response = user_response.lower()
  if(user_response != 'bye'):
    if(user_response == 'thankyou' or user_response == 'thanks'):
      flag = False
      print('Bot: You are welcome.')
    else:
      if(greet(user_response) != None):
        print('Bot:' + greet(user_response))
      else:
        sentence_tokens.append(user_response) # sentence tokens will generate
        word_tokens = word_tokens + nltk.word_tokenize(user_response) # word token will generate
        final_words = list(set(word_tokens)) # final words will get choosen
        print('Bot:',end ='')
        print(response(user_response)) # responce will be generated
        sentence_tokens.remove(user_response)

  else:
    flag = False
    print('Bot: Goodbye!')