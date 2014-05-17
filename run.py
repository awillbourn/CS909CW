import data_import
import represent
import pickle
import classify
import nltk

# setup
dir = 'C:/Users/Andrew/Dropbox/CS909/CW/reuters21578/'
relevant_topics = ("earn", "acquisitions","money-fx","grain","crude","trade","interest","ship","wheat","corn")



# get raw data
print "importing data"
#topics, bodies, training_index = data_import.getTopics_Bodies(dir, relevant_topics)
print "import done"


##******************pickle dump / load section***************

##    dump raw data
##
#pickle.dump(topics, open("save_topics.p", "wb"))
#pickle.dump(bodies, open("save_bodies.p", "wb"))
#pickle.dump(training_index, open("save_tindex.p", "wb"))

####      load raw Data
##
topics = pickle.load(open("save_topics.p", "rb"))
bodies = pickle.load(open("save_bodies.p", "rb"))
training_index = pickle.load(open("save_tindex.p", "rb"))
##

##***********************************************************



# get uni, bi, trigrams
print "getting unigram, bigram, trigram representations"
#unigram, bigram, trigram = represent.getGrams(bodies)

#get Features

print "getting Topic model features..."
#Tfeatures = represent.getTopicModel(bodies, 100)
print "getting uni, bi, trigrams..."
print "getting features for uni, bi, trigrams..."

print "unigram..."
#Ufeatures = represent.getTopicModel(unigram, 100)
print "bigram..."
#Bfeatures = represent.getTopicModel(bigram, 80)
print "trigram..."
#Trifeatures = represent.getTopicModel(trigram, 50)



##    dump features

#pickle.dump(Tfeatures, open("save_Tfeatures.p", "wb"))
#pickle.dump(Ufeatures, open("save_Ufeatures.p", "wb"))
#pickle.dump(Bfeatures, open("save_Bfeatures.p", "wb"))
#pickle.dump(Trifeatures, open("save_Trifeatures.p", "wb"))

##
####    load features
##
Tfeatures = pickle.load(open("save_Tfeatures.p", "rb"))
Ufeatures = pickle.load(open("save_Ufeatures.p", "rb"))
Bfeatures = pickle.load(open("save_Bfeatures.p", "rb"))
Trifeatures = pickle.load(open("save_Trifeatures.p", "rb"))

##****************** pickle done *********************


print "building test and train data"

Ttrain, Ttest = classify.GetTrainTest(Tfeatures, topics, training_index)
print "Topics test NB"
#classify.Train(Ttrain, relevant_topics, 0)
print "Topics test Tree"
#classify.Train(Ttrain, relevant_topics, 1)

Utrain, Utest = classify.GetTrainTest(Ufeatures, topics, training_index)
print "Unigrams test NB"
#classify.Train(Utrain, relevant_topics, 0)
print "Unigrams test Tree"
#classify.Train(Utrain, relevant_topics, 1)

Btrain, Btest = classify.GetTrainTest(Bfeatures, topics, training_index)
print "Bigrams test NB"
#classify.Train(Btrain, relevant_topics, 0)
print "Bigrams test Tree"
#classify.Train(Btrain, relevant_topics, 1)

Tritrain, Tritest = classify.GetTrainTest(Trifeatures, topics, training_index)
print "Trigrams test NB"
#classify.Train(Tritrain, relevant_topics, 0)
print "Trigramss test Tree"
#classify.Train(Tritrain, relevant_topics, 1)

#best classifer is NaiveBayes over the Unigrams set, train and run on test data, collect results

print "training final classifier - NaiveBayes over thh Unigrams set"
classifier = nltk.classify.NaiveBayesClassifier.train(Utrain)

print "running over test data"
acc = nltk.classify.accuracy(classifier, Utest)


TP = list([0]*10)
FP = list([0]*10)
FN = list([0]*10)
PRE = list([0]*10)
REC = list([0]*10)

REC_Macro = 0
REC_Micro = 0

PRE_Macro = 0
PRE_Micro = 0


for k in Utest:
    obs_class = classifier.classify(k[0])
    if (obs_class == k[1]):
        TP[relevant_topics.index(k[1])] +=1
    else:
        FP[relevant_topics.index(k[1])] +=1
        FN[relevant_topics.index(obs_class)] +=1

for i in range(0,10):
    if TP[i] + FN[i] == 0:
        REC[i] = 0
    else:
        REC[i] = TP[i] / float(TP[i] + FN[i])
    if TP[i] + FP[i] == 0:
        PRE[i] = 0
    else:
        PRE[i] = TP[i] / float(TP[i] + FP[i])

PRE_Macro = sum(PRE) / float(10)
REC_Macro = sum(REC) / float(10)

PRE_Micro = sum(TP) / float( sum(TP) + sum(FP) )
REC_Micro = sum(TP) / float( sum(TP) + sum(FN) )


print "Accuracy: "
print acc

print "TP:"
print TP
print "FN"
print FN
print "FP"
print FP

print "recall: "
print REC
print "precision: "
print PRE

print "precision macro: "
print PRE_Macro
print "recall macro: "
print REC_Macro
print "precision micro: "
print PRE_Micro
print "recall micro: "
print REC_Micro


