import torch as tr
import collections
import numpy as np
import os
import pandas as pd
from sklearn import preprocessing
from torchnlp.encoders.text import StaticTokenizerEncoder, stack_and_pad_tensors
from torchnlp.encoders.text.default_reserved_tokens import DEFAULT_PADDING_INDEX
import re
import matplotlib.pyplot as plt

TRAIN_PATH = r"data/train/"
TEST_PATH = r"data/test/"
POS_FOLDER = "positive"
NEG_FOLDER = "negative"
PRETRAIN_FILE = r"data/all.review.vec.txt"

# suggested parameters
TOP_WORD_INDEX = 10000
EMBED_VEC_SIZE = 100


def read_data_into_df(path, folder):
    """
    Function to read data into dataframe
    :param path: string of path location
    :param folder: folder name, whether positive or negative
    :return: dataframe of review data into one table
    """
    data_dic = {}
    for file in os.listdir(path+folder):
        review = open(path+folder+"/"+file).read()
        data_dic[file] = review

    data_df = pd.DataFrame.from_dict(data=data_dic, orient='index', columns=['review']).reset_index()
    # adding indicator for positive or negative
    data_df['sentiment'] = 1 if (folder == 'positive') else 0

    return data_df.reset_index(drop=True)


def read_all_sentiment_data(path, regex=False):
    """
    Aggregate function that combine pos and neg data
    """
    pos_df = read_data_into_df(path, POS_FOLDER)
    neg_df = read_data_into_df(path, NEG_FOLDER)
    data_df = pd.concat([pos_df, neg_df]).reset_index(drop=True)
    if regex:
        # clean up the input review
        data_df.review = data_df.review.apply(remove_special_char)
    return data_df

def explore_data(path):
    """
    For Question 1, explore the data metrics and print out information
    :param path: path location for the training data
    :return: N/A
    """
    data_df = read_all_sentiment_data(path)
    # calculate word count per row
    data_df['wordcount'] = [len(s.split()) for s in data_df.review.tolist()]
    # get unique word count
    unique_word_count = len(set(" ".join(data_df.review).split()))
    pos_neg_ratio = data_df[data_df.sentiment == 1].shape[0] / data_df[data_df.sentiment == 0].shape[0]
    avg_doc_len = data_df.wordcount.mean()
    max_doc_len = data_df.wordcount.max()
    # printing our result
    print("The total number of unique words in T: {}".format(unique_word_count))
    print("The total number of training examples in T: {}".format(data_df.shape[0]))
    print("The ratio of positive examples to negative examples in T: {}".format(pos_neg_ratio))
    print("The average length of document in T: {}".format(avg_doc_len))
    print("The max length of document in T: {}".format(max_doc_len))


def remove_special_char(string):
    """
    Helper function to pre-process data by removing special character and punctuation
    :param string: input review sentence
    :return: review string after removing special characters and unneeded space, also turn into lower case
    """
    s = re.sub('[^a-zA-Z0-9 ]', "", string)
    s = re.sub(r' +', " ", s)
    s = re.sub(r'\n', " ", s)
    s = s.lower()

    return s


def mod_pad_tensor(tensor, length, padding_index=DEFAULT_PADDING_INDEX):
    """
    Modified version of pytorch pad_tensor function, to handle input tensor longer than length case
    Args:
        tensor (torch.Tensor [n, ...]): Tensor to pad.
        length (int): Pad the ``tensor`` up to ``length``.
        padding_index (int, optional): Index to pad tensor with.

    Returns
        (torch.Tensor [length, ...]) Padded Tensor.
    """
    n_padding = length - tensor.shape[0]
    if n_padding == 0:
        return tensor
    if n_padding < 0:
        # instead of taking first length many words, using the last length many words
        return tensor[tensor.shape[0]-length:]
    # if we add padding, to keep consistency , we add the padding at first rather than last
    padding = tensor.new(n_padding, *tensor.shape[1:]).fill_(padding_index)
    return tr.cat((tensor, padding), dim=0)


def create_data_encoder(data_df):
    """
    Function to create an encoder for data pre-processing
    :param data_df: raw dataframe storing all input information
    :return: encoder to encoder the review, and a dictionary saving the most frequently appear K words
    """
    # initialize encoder first
    encoder_init = StaticTokenizerEncoder(data_df.review, tokenize=lambda s: s.split())
    # take the word count
    tokens = encoder_init.tokens.items()
    # sort top token by word count in reverse order (largest first)
    sorted_tokens = collections.OrderedDict(sorted(tokens, key=lambda p : p[1], reverse=True))
    # NOTE that StaticTokenizerEncoder reverses 0-4 for five special char, so we could only keep the top K -5
    # in order to preserve the index to be consistent
    top_tokens = list(sorted_tokens.keys())[0:TOP_WORD_INDEX-5]
    # to form top dic, need to insert the 5 words to preserve embedding dictionary index
    top_token_idx_dic = {k: v for v, k in enumerate(['<pad>', '<unk>', '</s>', '<s>', '<copy>']+top_tokens)}

    # we need to initialize a new encoder using the word frequency, so that the word embedding index
    # could corresponding to the embedding matrix index - this is because statictokenizerencoder
    # encode based on appearance order but not frequency order
    encoder = StaticTokenizerEncoder([" ".join(t for t in top_tokens)], tokenize=lambda s: s.split())

    return encoder, top_token_idx_dic


def data_preprocess(encoder, data_df):
    """
    Pre-process data using the designed encoder
    :param encoder: pytorch encoder object with word index based on word frequency
    :param data_df: input data (not yet pre-processed)
    :return:
        nd-array of padded sequence for each review, dimension is #review x 100
        nd-array of sentiment value for each review
    """
    # pad tokens of text - each is a tensor item
    encoded_data = [encoder.encode(review) for review in data_df.review]
    # use mod_pad_tensor to restrict the length
    encoded_pad_data = [mod_pad_tensor(x, length=EMBED_VEC_SIZE) for x in encoded_data]
    # stak all pad tensor into nd-array
    encoded_pad_data = stack_and_pad_tensors(encoded_pad_data)
    # sentiment encoding
    # for simply 0 and 1 value, no need to use pytorch encoder
    stmt_encoder = preprocessing.LabelEncoder()
    encoded_sentiment = stmt_encoder.fit_transform(data_df.sentiment)

    return encoded_pad_data.tensor.numpy(), encoded_sentiment


def load_pretrain_data(file=PRETRAIN_FILE):
    """
    Function to load the pre-trained embedding vectors and save in dictionary
    """
    pretrain_dic = {}
    data = open(file)
    for i, vec in enumerate(data):
        vec_val = vec.split()
        # first element is the corresponding word
        word = vec_val[0]
#        pretrain_dic[word] = vec_val[1:]
        # each value is a np array
        pretrain_dic[word] = np.asarray(vec_val[1:])

    return pretrain_dic


def create_embed_mtx(top_token_idx_dic):
    """
    Function to create the nd-matrix of embedding matrix
    """
    pretrain_dic = load_pretrain_data()
    # initialize matrix with zeros
    embedding_matrix = np.zeros((TOP_WORD_INDEX, EMBED_VEC_SIZE))
    in_embedding = 0
    for word, idx in top_token_idx_dic.items():
        if word in pretrain_dic.keys():
            embedding_matrix[idx] = pretrain_dic[word]
            in_embedding += 1
        #else:
            #embedding_matrix[idx] = np.random.normal(scale=0.6, size=(EMBED_VEC_SIZE,))
    print("# of tokens appearing in pre-trained data is {}".format(in_embedding))
    return embedding_matrix


def plot_model_result(fit_model, model_name, metric):
    """
    Helper function to plot the loss or accurancy by epoches
    """
    plt.figure(model_name+" "+metric)
    plt.plot(fit_model.history[metric.lower()], color='skyblue')
    plt.plot(fit_model.history['val_'+metric.lower()], color='purple')
    plt.title(model_name+" "+metric)
    plt.ylabel(metric)
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
