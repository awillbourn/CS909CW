import gensim, logging, bz2
import nltk


    #create the topic model
def getTopicModel(text, NoTopics):
    #build dictionary
    diction = gensim.corpora.Dictionary(text)

    #get corpus
    cor = [diction.doc2bow(t) for t in text]
    
    #create tfidf model
    model = gensim.models.TfidfModel(cor)
    tfidf_m = model[cor]
    model = gensim.models.LdaModel(tfidf_m, id2word=diction, num_topics=NoTopics)

    # get the actual feature set

    tf = model[tfidf_m]

    # round values to 1 decimal places and return feature set

    features = list()

    for f in tf:
        temp = {}
        for k in range(0,len(f)):
            temp[f[k][0]] = round(f[k][1],1)
        features.append(temp)
        
    return features

def getGrams(tokens):
    from nltk import bigrams
    from nltk import trigrams
    
    #remove tokens that are within the top 50 frequency rating and past 2000
    print "filtering..."
    fdist = nltk.FreqDist()
    for t in tokens:
        fdist.update(nltk.FreqDist(t))

    common = fdist.keys()[5:500]

    filtered = list()

    for i in range(0, len(tokens)):
        filtered.append([t for t in tokens[i] if t in common])

    # we have our unigrams
    uni = filtered

    print "getting bigrams..."

    # build bigrams
    bi = list()
    for t in filtered:
        temp = (bigrams(t))
        temp[:] = [str(x) for x in temp]
        bi.append(temp)
        


    print "getting trigrams..."
        
    #build trigrams
    tri = list()
    for t in filtered:
        temp = (trigrams(t))
        temp[:] = [str(x) for x in temp]
        tri.append(temp)

    
    return uni, bi, tri
