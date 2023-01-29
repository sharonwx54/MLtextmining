from data_preprocess import *
from scipy.sparse import csr_matrix

TOKEN_PICKLE = "top2000{}token.pickle"
VEC_PICKLE = "{}_feature_{}.pickle"
IDF_PICKLE = "allIDFs.pickle"


# CTF/DF feature matrix
def create_feature_mtx(data_dic, feature_type, data_type):
    # function to create the feature matrix by tf based on token selection
    from_pkl = TOKEN_PICKLE.format(feature_type)
    to_pkl = VEC_PICKLE.format(feature_type, data_type)
    texts = data_dic['text']
    with open(from_pkl, 'rb') as r:
        freq_dic = pickle.load(r)
        freq_list = list(freq_dic.keys())
    vec_list, row_idx, col_idx = [], [], []
    for r in range(len(texts)):
        text = texts[r]
        vec_idx = [freq_list.index(w) for w in text if w in freq_list]
        vec = [text.count(freq_list[idx]) for idx in set(vec_idx)]
        vec_list.extend(vec)
        col_idx.extend(list(set(vec_idx)))
        row_idx.extend([r]*len(set(vec_idx)))
    # save as a sparse matrix to save memory
    vec_csr = csr_matrix((vec_list, (row_idx, col_idx)))
    with open(to_pkl, 'wb') as pkl:
        pickle.dump(vec_csr, pkl, protocol=pickle.HIGHEST_PROTOCOL)

    return vec_csr


def feature_engineer(data_dic, data_type):
    """
    for feature engineering, we add idf where idf is log(N+1)/df
    however, instead of using idf to select token, we use DF token as it performs the best
    for each feature value, instead of using tf count, we use 0.9*tf + 0.05*idf + 0.05/CTF for each token
    """
    # loading DF and CTF token frequency
    from_pkl_df = TOKEN_PICKLE.format("DF")
    from_pkl_ctf = TOKEN_PICKLE.format("CTF")
    to_pkl = VEC_PICKLE.format("ENG", data_type)
    texts = data_dic['text']
    with open(from_pkl_df, 'rb') as r:
        freq_dic = pickle.load(r)
        freq_list = list(freq_dic.keys())
    with open(from_pkl_ctf, 'rb') as r_ctf:
        ctf_dic = pickle.load(r_ctf)
        ctf_sub_dic = {}
        # create a sub dictionary for CTF, since not every word in DFfreq appears in CTFfreq
        # so we need to handle 0 appearance case. Assign the smallest possible count 1
        for k in freq_list:
            ctf_sub_dic[k] = ctf_dic[k] if k in ctf_dic.keys() else 1
        # ctf_sub_dic = {ctf_dic[k] if k in ctf_dic.keys() else k:1 for k in freq_list}
    with open(IDF_PICKLE, 'rb') as r_idf:
        idf_dic = pickle.load(r_idf)
    vec_list, row_idx, col_idx = [], [], []

    for r in range(len(texts)):
        text = texts[r]
        vec_idx = [freq_list.index(w) for w in text if w in freq_list]
        # we use 90% tf, 5% idf value (to penalize high df) and 5% inverse CTF value (to penalize high ctf)
        vec = [0.9*text.count(freq_list[idx])+0.05*idf_dic[freq_list[idx]]+0.05/ctf_sub_dic[freq_list[idx]] for idx in set(vec_idx)]
        vec_list.extend(vec)
        col_idx.extend(list(set(vec_idx)))
        row_idx.extend([r]*len(set(vec_idx)))
    vec_csr = csr_matrix((vec_list, (row_idx, col_idx)))

    with open(to_pkl, 'wb') as pkl:
        pickle.dump(vec_csr, pkl, protocol=pickle.HIGHEST_PROTOCOL)
    return vec_csr


def run_feature_creation():
    # aggregate function to create all feature matrix at once
    # save to pickle for feature easy access
    with open("train_data.pickle", 'rb') as train_r:
        train_dic = pickle.load(train_r)
        create_feature_mtx(train_dic, "CTF", 'train')
        create_feature_mtx(train_dic, "DF", 'train')
        feature_engineer(train_dic, 'train')
        # create_feature_mtx(train_dic, "TFIDF", 'train')
    with open("dev_data.pickle", 'rb') as dev_r:
        dev_dic = pickle.load(dev_r)
        create_feature_mtx(dev_dic, "CTF", 'dev')
        create_feature_mtx(dev_dic, "DF", 'dev')
        feature_engineer(dev_dic, 'dev')
    with open("test_data.pickle", 'rb') as test_r:
        test_dic = pickle.load(test_r)
        create_feature_mtx(test_dic, "CTF", 'test')
        create_feature_mtx(test_dic, "DF", 'test')
        feature_engineer(test_dic, 'test')
    print("Finishing creating feature matrix for all dataset")
