# 11.641 - HW1
# Author - Sharon (Wenxin) Zhang
# AndrewID - wenxinz3

# Imports
import pandas as pd
import numpy as np
from scipy import sparse
import time
import os

# Global Variables
DAMPENING_FACTOR = 0.8
BETA = 0.15
CONVERGE_THRESHOLD = 1e-8
TRANSITION_PATH = r"data/transition.txt"
DOC_TOPIC_PATH = r"data/doc_topics.txt"
QTSPR_DISTRI_PATH = r"data/query-topic-distro.txt"
PTSPR_DISTRI_PATH = r"data/user-topic-distro.txt"
INDRI_PATH = r"./data/indri-lists"
OUTPUT_PATH = r"./"

GPR_TXT_FILE = r"GPR.txt"
QTSPR_TXT_FILE = r"QTSPR.txt"
PTSPR_TXT_FILE = r"PTSPR.txt"

COMBINE_METHOD_WT_MAP = {'NS': (0.0, 1.0), 'WS': (0.2, 0.8), 'CM': (0.9, 0.1)}
SEARCH_SCORE_COL_NAMES = ['QueryID', 'Q0', 'DocID', 'Rank', 'Score', 'RunID']


# Global PageRank
def global_PR(r_vec, trans_mtx, zero_outlink_idx, alpha):
    """
    Function to create GPR vector given initial r vector, transition matrix and alpha
    Note that zero_outlink_idx is a list of doc/node indices that have 0 links
    """
    # start_time = time.time()
    doc_num = trans_mtx.shape[0]
    p_0 = np.vstack([1.0 / doc_num] * doc_num)
    zero_outlink_vector = update_zero_outlink_vec(zero_outlink_idx, r_vec, init=True)
    converge = False
    iter_num = 0
    while not converge:
        r_vec_next = (1 - alpha) * (trans_mtx.transpose().dot(r_vec)) + (1 - alpha) * zero_outlink_vector + alpha * p_0
        # record 1 iteration, calculate difference between curr and prev r,
        # and update r_vec to be the r_vec next for next iteration
        iter_num += 1
        r_diff = np.linalg.norm((r_vec_next - r_vec), ord=1)
        r_vec = r_vec_next
        # if the distance between r_k-1 and r_k is small enough, we reach convergence
        if r_diff < CONVERGE_THRESHOLD:
            converge = True
        # if not yet converge, we need to update the zero_outlink vec for next iteration,
        # as it is subject to the r_vec
        zero_outlink_vector = update_zero_outlink_vec(zero_outlink_idx, r_vec)

    # end_time = time.time()

    # print("GPR PR Time: " + "--- %s seconds ---" % (end_time - start_time))
    # print("Total Iteration: {}".format(iter_num))

    return r_vec


# TSPR link score calculation
def link_score_TSPR(topic_r_vec, topic_prob_vec, trans_mtx, zero_outlink_idx, alpha, beta):
    """
    Function to create only the offline converging link-based score per topic,
    given initial r vector per topic, transition matrix,
    query-topic or user-topic probability vector per query and alpha, beta
    Note that zero_outlink_idx is a list of doc/node indices that have 0 links
    """
    # start_time = time.time()
    # initialize p_0 and zero_outlink vector
    doc_num = trans_mtx.shape[0]
    p_0 = np.vstack([1.0 / doc_num] * doc_num)
    zero_outlink_vector = update_zero_outlink_vec(zero_outlink_idx, topic_r_vec, init=True)
    converge = False
    iter_num = 0

    while not converge:
        topic_r_vec_next = alpha * (trans_mtx.transpose().dot(topic_r_vec)) + alpha * zero_outlink_vector \
                           + beta * topic_prob_vec + (1 - alpha - beta) * p_0
        # record 1 iteration and update r_vec to be the r_vec next for next iteration
        iter_num += 1
        r_diff = np.linalg.norm((topic_r_vec_next - topic_r_vec), ord=1)
        topic_r_vec = topic_r_vec_next
        # if the distance between r_k-1 and r_k is small enough, we reach convergence
        if r_diff < CONVERGE_THRESHOLD:
            converge = True
        # if not yet converge, we need to update the zero_outlink vec for next iteration,
        # as it is subject to the r_vec
        zero_outlink_vector = update_zero_outlink_vec(zero_outlink_idx, topic_r_vec)
    # print("Total Iteration: {}".format(iter_num))
    # end_time = time.time()
    # print("TSPR PR Time: " + "--- %s seconds ---" % (end_time - start_time))
    return topic_r_vec


# TSPR combined content and link score calculation per query/user
def full_score_TSPR(doc_topic_mtx, item_topic_distri, trans_mtx, zero_outlink_idx, alpha, beta):
    """
    Function to calculated TSPR vector that combines content and link score calculation per query/user
    given entire doc-topic matrix, transition matrix,
    query-topic or user-topic probability vector per query and alpha, beta

    Returns: an n x q matrix, where n is number of document and q is the number of user-query pairs
    """
    # start_time = time.time()
    # retrieve topic and document number
    doc_num, topic_num = doc_topic_mtx.shape
    p_0 = np.vstack([1.0 / doc_num] * doc_num)
    # for each topic, iterate until convergence for the offline link based r_t
    rT_mtx_cols = []
    for t in range(0, topic_num):
        rt_vec_init = p_0
        topic_prob_vec = doc_topic_mtx[:, t]
        # print("Iteration for topic: {}".format(t))
        rt_vec_stable = link_score_TSPR(
            rt_vec_init, topic_prob_vec, trans_mtx, zero_outlink_idx, alpha, beta)
        rT_mtx_cols.append(rt_vec_stable)
    # end_time = time.time()
    # print("Total TSPR: " + "--- %s seconds ---" % (end_time - start_time))
    # with a matrix of n x T, we calculate the weighted sum of TSPR using item topic distribution
    # NOTE the item is a combo of user and query, and the ids do not align w indexing
    # convert list of vector into matrix before doc product
    rT_mtx = np.array(rT_mtx_cols).reshape(doc_num, topic_num)
    # q x n matrix, each row is r_q, in the order of distri_index
    TSPR_mtx = item_topic_distri.to_numpy().dot(rT_mtx.transpose())
    return TSPR_mtx.transpose()  # n x q


def combine_IR_PR_per_query_user(pr_vec, ir_df, ir_wt, pr_wt, if_cm=False):
    """
    Function to incorporate search IR scores into PR score based on different weighting method
    :param pr_vec: pr_vec for TSPR would be per query-user based, vector of probabilities
    :param ir_df: comes from each txt file in indri-lists - per query user based as well
    :param ir_wt: float, weighting on IR scores
    :param pr_wt: float, weighting on PR scores
    :param if_cm: boolean to detect if the weighting method is CM
    :return: DataFrame of ranking score recalculated using weights on IR and PR score, scored in ascending order
    """
    # NOTE that pr_vec is in order by docID, but search_score_df is not and would be sparse
    in_ir_doc_pr = pr_vec[np.r_[ir_df.DocID - 1], :]
    ir_df['PRScore'] = pd.Series(in_ir_doc_pr.transpose()[0])
    # if the method is CM, we use log function on PR score to convert it to negative value for
    # comparison w search scores, which are negative
    if if_cm:
        ir_df.PRScore = np.log(ir_df.PRScore)
    ir_df['Score'] = ir_df.Score * ir_wt + ir_df.PRScore * pr_wt
    # rerank the query-user scores by combined scores
    ir_df['Rank'] = ir_df.Score.rank(method='dense', ascending=False)

    # sort by the new Ranking in ascending order
    ir_df = ir_df.sort_values(by=['Rank']).drop(columns=['PRScore'])
    return ir_df


def run_TSPR_into_file(tspr_type, tspr_mtx, query_user_idx, ir_score_df, wt_method):
    """
    Function to run and calculate final TSPR score for each query-user pairs and save all to one txt
    file for trec_eval valuation
    :param tspr_type: PTSPR or QTSPR, mainly for naming purpose
    :param tspr_mtx: TSPR matrix for purely PR scores, for all query-user pairs
    :param query_user_idx: this is the mapping table for index to user-query pairs,
        generally the same for PTSPR and QTSPR, but still taking this argument to make the function generic
    :param ir_score_df: the search IR scores all combined into one txt, from indri-lists directory
    :param wt_method: NS, WS, or CM
    :return: NULL, but write the TSPR ranking into txt
    """
    # for each user-query tspr score vector, we combine with ir scores provided and write into
    # txt file
    ir_wt, pr_wt = COMBINE_METHOD_WT_MAP[wt_method]
    full_tspr_files = []
    for i in range(0, tspr_mtx.shape[1]):
        # start_time = time.time()
        qry_user_id = "{}-{}".format(query_user_idx.UserID[i], query_user_idx.QueryID[i])
        qry_user_ir_df = ir_score_df[ir_score_df.QueryID == qry_user_id]
        r_qry_user = tspr_mtx[:, i].reshape((-1, 1))
        comb_score_df = combine_IR_PR_per_query_user(r_qry_user, qry_user_ir_df, ir_wt, pr_wt, wt_method == 'CM')
        full_tspr_files.append(comb_score_df)
        # end_time = time.time()
        # print("TSPR Retrieval Time for {} ".format(i) + "--- %s seconds ---" % (end_time - start_time))

    full_tspr_df = pd.concat(full_tspr_files)
    full_tspr_df.to_csv(OUTPUT_PATH + "/" + tspr_type + "-" + wt_method + ".txt", sep=" ", index=False, header=False)


def run_GPR_into_file(gpr_vec, ir_score_df, wt_method):
    """
    Function to run calculate final TSPR score for each query-user pairs and save all to one txt
    file for trec_eval valuation
    :param tspr_type: PTSPR or QTSPR, mainly for naming purpose
    :param tspr_mtx: TSPR matrix for purely PR scores, for all query-user pairs
    :param query_user_idx: this is the mapping table for index to user-query pairs,
        generally the same for PTSPR and QTSPR, but still taking this argument to make the function generic
    :param ir_score_df: the search IR scores all combined into one txt, from indri-lists directory
    :param wt_method: NS, WS, or CM
    :return: NULL, but write the TSPR ranking into txt
    """
    ir_wt, pr_wt = COMBINE_METHOD_WT_MAP[wt_method]
    full_grp_files = []
    query_user_ids = ir_score_df.QueryID.unique().tolist()
    for i in range(0, len(query_user_ids)):
        start_time = time.time()
        qry_user_id = query_user_ids[i]
        qry_user_ir_df = ir_score_df[ir_score_df.QueryID == qry_user_id]
        comb_score_df = combine_IR_PR_per_query_user(gpr_vec, qry_user_ir_df, ir_wt, pr_wt)
        full_grp_files.append(comb_score_df)
        print("GPR Retrieval Time: " + "--- %s seconds ---" % (end_time - start_time))
    full_gpr_df = pd.concat(full_grp_files)
    full_gpr_df.to_csv(OUTPUT_PATH + "/GPR-" + wt_method + ".txt", sep=" ", index=False, header=False)


def write_pr_into_file(pr_vec, pr_name):
    """
    Function to write pageRank score into txt file on doc-pr score manner
    :param pr_vec: pageRank vector
    :param pr_name: pageRank algorithm type, GPR, QTSPR or PTSPR

    """
    pr_vec_in_df = pd.DataFrame(pr_vec, columns=['PageRankValue'])
    pr_vec_in_df.index += 1
    pr_vec_in_df.index.name = 'documentID'

    pr_vec_in_df.reset_index().to_csv(
        OUTPUT_PATH + "/{}.txt".format(pr_name), sep=" ", index=False, header=False, float_format='%.8g')


# Other Helper Functions
def read_create_transition_mtx_from_file(file_path):
    """
    Function to read transition matrix from file into a sparse COO matrix under scipy
    Since the matrix size is large but sparse, COO structure is a good fit from efficiency perspective
    :param file_path: string of the file name
    :return:
        transition matrix under COO matrix type
        list of document index (docid) that which the doc no links to any other node
    """
    matrix = pd.read_csv(file_path, sep=" ", header=None)
    matrix.columns = ["Row", "Col", "Val"]
    max_shape = max(matrix.Row.max(), matrix.Col.max())
    # for each node, sum unique column get # of links the node out-link
    outlink_df = matrix.groupby("Row")[["Col"]].nunique()
    outlink_df.Col = 1.0 / outlink_df.Col
    matrix["Normalized_Val"] = matrix.Row.map(outlink_df.Col)

    # Note all the row and column index should subtract 1 for indexing purpose
    matrix.Row = matrix.Row - 1
    matrix.Col = matrix.Col - 1

    outlink_transition_mtx = sparse.coo_matrix(
        (matrix.Normalized_Val, (matrix.Row, matrix.Col)), shape=(max_shape, max_shape))
    # Note that above transition matrix only handle nodes with outlinks
    # In probability transition matrix, we need to add 1/n to all nodes without any links
    # However, given the nature of PageRank convergence formula,
    # we could decompose the pt-matrix into the sum of two matrices
    # 1. pure transition matrix for nodes with outlinks (transition matrix above)
    # 2. pure matrix that have 1/n for nodes without any outlink, and 0 for nodes with outlinks
    # On 2. when its transpose dot product with r_k, we get 1/n * (r_x, r_x, ...)n where r_x is
    # the sum of all r_i for all i that is a node without outlink
    # In case when we evenly distribute r_0 to be 1/n for each vector value, we would have
    # a vector with n element, each is 1/n * (# node without any outlink / # total nodes)

    zero_outlink_idx = [n - 1 for n in range(1, max_shape) if n not in outlink_df.index]

    return outlink_transition_mtx, zero_outlink_idx


def read_doc_topic_distri_from_file(file_path):
    """
    Function to read doc-topic probability distribution from file
    :param file_path: doc_topic.txt
    :return: CSC matrix n by T, where each column is a topic
    """
    matrix = pd.read_csv(file_path, sep=" ", header=None)
    matrix.columns = ["DocID", "TopicID"]
    # get document number and topic number
    doc_num = matrix.DocID.max()
    topic_num = matrix.TopicID.max()
    topic_doc_df = matrix.groupby("TopicID")[["DocID"]].nunique()
    # normalized value by number of doc per topic
    topic_doc_df.DocID = 1.0 / topic_doc_df.DocID
    matrix["Normalized_Val"] = matrix.TopicID.map(topic_doc_df.DocID)

    # Note all the row and column index should subtract 1 for indexing purpose
    matrix.TopicID = matrix.TopicID - 1
    matrix.DocID = matrix.DocID - 1
    # use CSC here for accessing columns by index
    doc_topic_mtx = sparse.csc_matrix(
        (matrix.Normalized_Val, (matrix.DocID, matrix.TopicID)), shape=(doc_num, topic_num))
    # return matrix is n x T, each column for one topic
    return doc_topic_mtx


def read_item_topic_distri_from_file(file_path):
    """
    Function to read query-topic or user-topic distribution files
    :param file_path: query/user-topic-distro.txt
    :return:
        DataFrame of query/user vs topic probability distribution, where
            index is user-query id pairs
            each row is for a user-query
            column is for each topic and is in Topic ID order
        DataFrame of an index to user-query ID match
            Technically this map is the same for query-topic or user-topic, but to keep consistency
            and generic, we return it for each call of the function
    """
    distri_df = pd.read_csv(file_path, sep=" ", header=None)
    # name column to match the topic ID, since default column idx start from 0,
    # and we have two non-topic columns before, we -1 to align topic 1 with name 1
    distri_df.columns = distri_df.columns - 1
    distri_df = distri_df.rename(columns={-1: "UserID", 0: "QueryID"})
    distri_df = distri_df.set_index(['UserID', 'QueryID']).sort_index()
    for col in distri_df.columns:
        # remove topic name to leave topic prob as cell value
        distri_df[col] = distri_df[col].str.split(':', expand=True)[[1]].rename(columns={1: col})
        distri_df[col] = distri_df[col].astype(float)

    # record the index to user/query pair mapping for accessing purpose
    distri_index_map = distri_df.index.to_frame(index=False)

    return distri_df, distri_index_map


def read_and_combine_search_results(file_dir_path):
    """
    Function to read from indri-lists folder all the search scores and combine into one table
    :param file_dir_path: indri-lists/
    :return: DataFrame of the search scores for each doc for each query-user pair
    """
    query_user_ir_df = []
    for file in os.listdir(file_dir_path):
        query_user_id = file.split('.')[0]
        ir_score_df = read_search_score_from_file(file_dir_path + "/" + file)
        ir_score_df.QueryID = query_user_id
        query_user_ir_df.append(ir_score_df)

    combine_ir_df = pd.concat(query_user_ir_df)
    return combine_ir_df


def read_search_score_from_file(file_path):
    """
    Function to read one user-query search score file into table
    :param file_path: indri-lists/user-query.results.txt
    :return: DataFrame of docID and all search related columns
    """
    search_score_df = pd.read_csv(file_path, sep=" ", header=None)
    search_score_df.columns = SEARCH_SCORE_COL_NAMES
    search_score_df.DocID = search_score_df.DocID.astype(int)

    return search_score_df


def update_zero_outlink_vec(zero_outlink_idx, r_vec, init=False):
    """
    Helper function to initialize or update the vector of probability transition matrix
    # for only nodes without outlinks, based on the current r_vec
    :param zero_outlink_idx: list of index representing doc that have 0 links to any other doc
    :param r_vec: vector of r - if this is an update, then the r_vec would r after previous iteration
    :param init: boolean, whether the function is called to initialize or update
    :return: n x 1 vector, each cell has identical value:
        if update, the value would be the sum of all current values for document
        without any outlinks divided by total doc number
        if initialize, the value would be the number of total doc without outlink over total doc number
        and then divided by total doc number
    """

    doc_num = len(r_vec)
    if init:
        zero_outlink_val = (len(zero_outlink_idx) / doc_num) / doc_num
    else:
        zero_outlink_val = 0
        for l in range(0, len(zero_outlink_idx)):
            zero_outlink_val += r_vec[zero_outlink_idx[l]]  # could be replaced by dot product
        zero_outlink_val = zero_outlink_val / doc_num

    zero_outlink_vec = np.vstack([zero_outlink_val] * doc_num)

    return zero_outlink_vec


def generate_sample_txt(GPR_vec, QTSPR_df, query_qu_idx, PTSPR_df, user_qu_idx):
    # Helpful function to generate sample txt for user 2 and query 1
    qu_query_idx = query_qu_idx[(query_qu_idx.UserID == 2) & (query_qu_idx.QueryID == 1)].index.values[0]
    qu_user_idx = user_qu_idx[(user_qu_idx.UserID == 2) & (user_qu_idx.QueryID == 1)].index.values[0]
    write_pr_into_file(GPR_vec, "GPR")
    qtspr_qu = QTSPR_df[:, qu_query_idx]
    ptspr_qu = PTSPR_df[:, qu_user_idx]
    write_pr_into_file(qtspr_qu, "QTSPR-U2Q1")
    write_pr_into_file(ptspr_qu, "PTSPR-U2Q1")

