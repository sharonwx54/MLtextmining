import string
import json
import pickle
import itertools
import collections
import numpy as np
import math

TRAIN_PATH = "data/yelp_reviews_train.json"
TEST_PATH = "data/yelp_reviews_test.json"
DEV_PATH = "data/yelp_reviews_dev.json"
STOPWORD_LIST = "data/stopword.list"
TOP_K_FOR_FEATURE = 2000
NUM_FEATURE = 5


def preprocess_text(text, stopwords):
    # process sentence
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.translate(str.maketrans('', '', string.digits))
    remain_list = [w for w in text.split() if w not in stopwords and w.isalpha()]

    return list(filter(None, remain_list))


def stars_to_mtx(star_list):
    # convert rating into a matrix of column 5
    star_ndarray = np.asarray(star_list).T
    # need to subtract one so as to align with the index
    star_mtx = np.eye(NUM_FEATURE)[star_ndarray-1]
    return star_mtx


def preprocess_train_into_dic(data, stopwords):
    # train_data.pickle
    data_dic = {'user_id':[], 'stars':[], 'text':[]}
    for d in data:
        data_dic["user_id"].append(d['user_id'])
        data_dic["stars"].append(d['stars'])
        processed_text = preprocess_text(d['text'], stopwords)
        data_dic["text"].append(processed_text)
    # save to pickle for further access
    with open('train_data.pickle', 'wb') as pkl:
        pickle.dump(data_dic, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    return data_dic


def preprocess_into_dic(data, filename, stopwords):
    # test_data/dev_data.pickle
    data_dic = {'user_id':[], 'text':[]}
    for d in data:
        data_dic["user_id"].append(d['user_id'])
        processed_text = preprocess_text(d['text'], stopwords)
        data_dic["text"].append(processed_text)
    # save to pickle for further access
    with open(filename+'.pickle', 'wb') as pkl:
        pickle.dump(data_dic, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    return data_dic


def get_data_stats(data_dic):
    # get CTF for token while printing out the stats
    tokens = itertools.chain.from_iterable(data_dic['text'])
    token_counter = collections.Counter(tokens)
    star_counter = collections.Counter(data_dic['stars'])
    print(token_counter.most_common(9))
    print(star_counter)
    total_star = len(data_dic['stars'])
    for k in dict(star_counter):
        print("Star {} has percentage: {}%".format(k, 100*star_counter[k]/total_star))
    # lastly save the corpus top 2000 word into pickle for ctf access
    with open('top'+str(TOP_K_FOR_FEATURE)+'CTFtoken.pickle', 'wb') as pkl:
        pickle.dump(dict(token_counter.most_common(TOP_K_FOR_FEATURE)), pkl, protocol=pickle.HIGHEST_PROTOCOL)


def get_token_df(data_dic, if_save=True, shift=0):
    # get DF frequency into dictionary, shift allows the window to slide from the top common word
    # to some later word with more information
    df_dic = {}
    for text in data_dic['text']:
        unique_tkn = set(text)
        for tkn in unique_tkn:
            if tkn in df_dic:
                df_dic[tkn]+=1
            else:
                df_dic[tkn]=1
    sorted_df_dic = sorted(df_dic.items(), key=lambda t: t[1], reverse=True)
    top_df_dic = dict(sorted_df_dic[shift:TOP_K_FOR_FEATURE+shift])
    if if_save:
        with open('top'+str(TOP_K_FOR_FEATURE+shift)+'DFtoken.pickle', 'wb') as pkl:
            pickle.dump(top_df_dic, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    return dict(sorted_df_dic)


def get_token_tfidf(data_dic):
    # get the tf*idf value for token, define idf as log((N+1)/df)
    tokens = itertools.chain.from_iterable(data_dic['text'])
    token_counter = collections.Counter(tokens)
    df_dic = get_token_df(data_dic, False)
    N = len(data_dic['text'])
    tfidf_dic = {}
    for t, df in df_dic.items():
        idf = math.log((N+1)/df)
        tfidf = idf*token_counter[t]
        tfidf_dic[t] = tfidf
    sorted_tfidf_dic = sorted(tfidf_dic.items(), key=lambda t: t[1], reverse=True)
    top_tfidf_dic = dict(sorted_tfidf_dic[:TOP_K_FOR_FEATURE])
    with open('top' + str(TOP_K_FOR_FEATURE) + 'TFIDFtoken.pickle', 'wb') as pkl:
        pickle.dump(top_tfidf_dic, pkl, protocol=pickle.HIGHEST_PROTOCOL)


def get_token_idf(data_dic):
    # saving the idf value for all tokens, not limited to most frequent words
    df_dic = get_token_df(data_dic, False)
    N = len(data_dic['text'])
    idf_dic = {}
    for t, df in df_dic.items():
        idf = math.log((N+1)/df)
        idf_dic[t] = idf
    with open('allIDFs.pickle', 'wb') as pkl_idf:
        pickle.dump(idf_dic, pkl_idf, protocol=pickle.HIGHEST_PROTOCOL)


def run_preprocessing():
    # aggregate function for data preprocessing
    # this function should only run once and all pickle files would be saved
    train_data = [json.loads(line) for line in open(TRAIN_PATH, 'r')]
    test_data = [json.loads(line) for line in open(TEST_PATH, 'r')]
    dev_data = [json.loads(line) for line in open(DEV_PATH, 'r')]

    with open(STOPWORD_LIST) as r:
        stopwords = r.read().split()
    print("Loading Train Data")
    train_dic = preprocess_train_into_dic(train_data, stopwords)
    print("Loading Dev Data")
    dev_dic = preprocess_into_dic(dev_data, "dev_data", stopwords)
    print("Loading Test Data")
    test_dic = preprocess_into_dic(test_data, "test_data", stopwords)

    # for train_data only, create token files for feature creation
    get_data_stats(train_dic)
    _ = get_token_df(train_dic)
    get_token_idf(train_dic)
    # get_token_tfidf(train_dic)


def svm_preprocessing(feature_type, data_type):
    # specific preprocessing for SVM as the function takes in data of different format
    with open('{}_feature_{}.pickle'.format(feature_type, data_type), 'rb') as r:
        data = pickle.load(r)
    writer = open("SVM_{}_{}.txt".format(feature_type, data_type), "w")
    print("Writing {} for SVM {} data".format(feature_type, data_type))
    for f in range(data.shape[0]):
        writer.write("0 ")
        for num, feature in zip(data[f].data, data[f].nonzero()[1]):
            writer.write("{}:{} ".format(feature+1, num))
        writer.write("\n")
    writer.close()


def svm_train_preprocessing(feature_type):
    # specific preprocessing for SVM as the function takes in data of different format
    # for training data, need different handling on labels
    with open('{}_feature_train.pickle'.format(feature_type), 'rb') as r:
        data = pickle.load(r)
    with open('train_data.pickle', 'rb') as data_train:
        train_dic = pickle.load(data_train)
        stars = train_dic['stars']

    writer = open("SVM_{}_train.txt".format(feature_type), "w")
    for f in range(data.shape[0]):
        writer.write("{} ".format(stars[f]))
        for num, feature in zip(data[f].data, data[f].nonzero()[1]):
            writer.write("{}:{} ".format(feature+1, num))
        writer.write("\n")
    writer.close()
