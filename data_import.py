import os.path
from bs4 import BeautifulSoup

from nltk import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import string


def getTopics_Bodies(input_dir, topics):
    #import the topics and bodies from the sgm files that have relevant topics
    dir = input_dir
    files = os.listdir(dir)

    #get text of the bodies
    body_text = []
    topic_text = []
    training_text = []
    
    print "importing data.."
    
    for f in files:
        if f.endswith("sgm") == True:
            soup = BeautifulSoup(open(dir+f))
            body_text.extend(soup.findAll('text'))
            topic_text.extend(soup.findAll('topics'))
            training_text.extend(soup.findAll('reuters'))

    print "tokenising and check relevent topics..."
    
    # tokenise the text and check that using instances of relevant topics
    # put all text to lowercase

    body_tok = []
    topic_tok = []
    training_list = []
    
    for i in range(0,len(body_text)):
        for t in topics:
            #check that it has atleast 1 relevant topic and has some text
            if (t in topic_text[i].prettify().split()) & (len(body_text[i]) > 0):        
                #convert to lowercase, and tokenise by space
                body_tok.append(nltk.word_tokenize(body_text[i].prettify().lower()))
                topic_tok.append(nltk.word_tokenize(topic_text[i].prettify().lower()))

                if training_text[i]['lewissplit'] == u"TRAIN":
                    training_list.append(1)
                else:
                    training_list.append(0)    
                
                break


    print "cleaning up tokens.."
    print "remove irrelevant topics, run stemmer, remove punctuation, remove stopwords, mark numbers as Num..."
    #clean up the tokens

    #remove irrelevant topics
    for i in range(0, len(topic_tok)):
        temp = []
        for t in topic_tok[i]:
            if t in topics:
                temp.append(t)
        topic_tok[i] = temp


    #build stemmer
    stemmer = SnowballStemmer("english", ignore_stopwords = True)
    
    #remove punctuation 
    punc = set(string.punctuation)
    for i in range (0, len(body_tok)):
        temp = []
        tokens = []
        for t in body_tok[i]:
            #remove all puncutation
            t = ''.join(ch for ch in t if ch not in set(string.punctuation)) 
            tokens.append(t)
            
        #run stemmer and remove stopwords from tokens
        tokens = [t for t in tokens if not t in stopwords.words("english")]
        tokens = [stemmer.stem(t) for t in tokens]
        
        #tag tokens which are numerical as NUM
        for k in range(0, len(tokens)):
            if tokens[k].isdigit():
                tokens[k] = u"NUM"
    
        body_tok[i] = tokens

        
        
    
    print "returning ", len(body_tok), " bodies"
    print "returning ", len(topic_tok), " topics"
    return topic_tok, body_tok, training_list


