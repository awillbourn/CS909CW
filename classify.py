import nltk
from random import randint

def GetTrainTest(features, topics, training_index):
    
    # build training and test set data
    f_trainData = list()
    f_testData = list()
    

    for i in range(0,len(training_index)):	
        if training_index[i] == 1:
            #add to training data
            # new instance for each possible correct classification
            
            for classification in topics[i]:
                f_trainData.append([features[i],classification])
        else:
            for classification in topics[i]:
                f_testData.append([features[i],classification])
    
    return f_trainData, f_testData

def Train(train, topics, cl):

    #setup k folds and run classifier.Return Avg accuracy, precision, recall

    trainData = list()

    # append which fold each peice of data is assigned randomly
    for i in range(0, len(train)):
        trainData.append([train[i], randint(0,9)])

    #variables for TP, FP, FN, accuracy, precision, recall for each classifer
    TP = (list([list([0]*10)]*10))
    FP = (list([list([0]*10)]*10))
    FN = (list([list([0]*10)]*10))
    ACC = list([0]*10)
    PRE = list([0]*10)
    REC = list([0]*10)

    AVG_TP = list([0]*10)
    AVG_FP = list([0]*10)
    AVG_FN = list([0]*10)

    AVG_ACC = 0

    REC_Macro = 0
    REC_Micro = 0

    PRE_Macro = 0
    PRE_Micro = 0


    # run folds
    for i in range(0,10):
        ktrain = [t[0] for t in trainData if t[1] != i]
        ktest = [t[0] for t in trainData if t[1] == i]

        if cl == 0:
            classifier = nltk.classify.NaiveBayesClassifier.train(ktrain)
        if cl == 1:
            classifier = nltk.classify.DecisionTreeClassifier.train(ktrain)
        
        ACC[i] = nltk.classify.accuracy(classifier, ktest)
        
        ## calculate TP, FP, FN lists for this classifier
        TP_temp = list([0]*10)
        FP_temp = list([0]*10)
        FN_temp = list([0]*10)
        for k in ktest:
            obs_class = classifier.classify(k[0])
            if (obs_class == k[1]):
                TP_temp[topics.index(k[1])] +=1
            else:
                FP_temp[topics.index(k[1])] +=1
                FN_temp[topics.index(obs_class)] +=1

        TP[i] = TP_temp
        FP[i] = FP_temp
        FN[i] = FN_temp

    print "TP values:"
    print TP
    print "FP values:"
    print FP
    print "FN values:"
    print FN

    #average out accuracy over classifiers
  
    AVG_ACC = sum(ACC) / 10

    #average out all TP, FP, FN over the classifiers.
    #the class
    for i in range(0,10):
        # the fold
        for k in range (0,10):
            AVG_TP[i] += TP[k][i]
            AVG_FP[i] += FP[k][i]
            AVG_FN[i] += FN[k][i]
        AVG_TP[i] = AVG_TP[i] / float(10)   #averaged out over the folds for each class
        AVG_FP[i] = AVG_FP[i] / float(10)
        AVG_FN[i] = AVG_FN[i] / float(10)
        
    # get recall and precision for each class 

    for i in range(0,10):
        if AVG_TP[i] != 0:
            REC[i] = AVG_TP[i] / float(AVG_TP[i] + AVG_FN[i])
            PRE[i] = AVG_TP[i] / float(AVG_TP[i] + AVG_FP[i])
        else:
            REC[i] = 0
            PRE[i] = 0

    # calculate macro stats (average over all classes)
    
    PRE_Macro = sum(PRE) / float(10)
    REC_Macro = sum(REC) / float(10)
    #calculate micro
    PRE_Micro = sum(AVG_TP) / float( sum(AVG_TP) + sum(AVG_FP) )
    REC_Micro = sum(AVG_TP) / float( sum(AVG_TP) + sum(AVG_FN) )

    print "Average TP: "
    print AVG_TP
    print "Average FP: "
    print AVG_FP
    print "Average FN: "
    print AVG_FN

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
    print "Average accuracy: "
    print AVG_ACC

    
                               
