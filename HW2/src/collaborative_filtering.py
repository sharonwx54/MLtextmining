import numpy as np

from src.utils import *
from sklearn.metrics.pairwise import cosine_similarity
import time

PCC_DEFAULT_DENOM = 1e-12
WTG_DEFAULT_DENOM = 1e-8


def top_n_neighbor_by_dot(csr_mtx, K, id_type, id=4321):
    if id_type == 'movie':
        csr_mtx = csr_mtx.T
    # grab the corresponding row vector of user / movie
    id_vec = csr_mtx[id, :]
    # dot product would return csr by default, this is a column of u or i by 1
    neighbor_mtx = csr_mtx.dot(id_vec.T)
    # take the single row vector out for sorting purpose
    id_vector = neighbor_mtx.T.toarray()[0]
    # we need to remove the user/movie itself as it would always be the closest
    # use mergesort to keep lower user id when tie happens
    id_vector_top_K = np.argsort(id_vector, kind='mergesort')[::-1][1:K+1]

    return id_vector_top_K, id_vector[id_vector_top_K]


def top_n_neighbor_by_cos(csr_mtx, K, id_type, id=4321):
    if id_type == 'movie':
        csr_mtx = csr_mtx.T
    id_vec = csr_mtx[id, :]
    # cosine_similarity function would return ndarray
    neighbor_mtx = cosine_similarity(csr_mtx, id_vec)
    # post cosine_similarity function, we get a ndarray directly
    id_vector = neighbor_mtx.T[0]
    # we need to remove the user/movie itself as it would always be the closest
    id_vector_top_K = np.argsort(id_vector, kind='mergesort')[::-1][1:K+1]

    return id_vector_top_K, id_vector[id_vector_top_K]


def memory_based_prediction(input_pair, csr_mtx, method=top_n_neighbor_by_cos, weight=False, K=10):
    movie_id, user_id = input_pair
    user_knn, user_knn_val = method(csr_mtx, K, "user", user_id)
    knn_movie_rating = csr_mtx[user_knn,:]
    if not weight:
        # for mean average
        avg_rating = knn_movie_rating[:, movie_id].toarray().mean()
    else:
        # for weighted average, need to use sum of absolute weights for denominator
        total_wtg = np.absolute(user_knn_val).sum()
        # avoid zero division
        if total_wtg == 0:
            total_wtg = WTG_DEFAULT_DENOM
        wtg_rating = user_knn_val.dot(knn_movie_rating[:, movie_id].toarray())[0]
        avg_rating = wtg_rating/total_wtg

    return avg_rating+3.0


def item_based_prediction(input_pair, csr_mtx, item_sim_mtx, weight=False, K=10):
    # technically this is the same method as memory-based, except
    # we compute the similarity matrix once and access it, rather than compute each time
    # this function in-take the computed similarity matrix
    movie_id, user_id = input_pair
    movie_similar_vec = item_sim_mtx[movie_id, :]
    movie_knn = np.argsort(movie_similar_vec, kind='mergesort')[::-1][1:K+1]
    movie_knn_val = movie_similar_vec[movie_knn]
    knn_movie_rating = csr_mtx[:, movie_knn]
    if not weight:
        # for mean average
        avg_rating = knn_movie_rating[user_id, :].toarray().mean()
    else:
        # for weighted average, need to use sum of absolute weights for denominator
        total_wtg = np.absolute(movie_knn_val).sum()
        # avoid zero division
        if total_wtg == 0:
            total_wtg = WTG_DEFAULT_DENOM
        wtg_rating = movie_knn_val.dot(knn_movie_rating[user_id, :].toarray().T)[0]
        avg_rating = wtg_rating/total_wtg

    return avg_rating+3


def standardize_mtx(input_mtx):
    # input could be single vector or matrix, always require calculation on row based
    avg_per_row = np.asarray(input_mtx.mean(axis=1))
    # subtract the mean value for each row, becoming np.matrix object
    standardize_mtx = input_mtx - avg_per_row
    # calculation (xi-x_mean)^2 for each cell
    sum_sqr_per_row = np.square(standardize_mtx).sum(axis=1)
    sqrt_sum_sqr_per_row = np.sqrt(sum_sqr_per_row)
    sqrt_sum_sqr_per_row[sqrt_sum_sqr_per_row == 0] = PCC_DEFAULT_DENOM
    standardize_mtx = standardize_mtx / sqrt_sum_sqr_per_row

    return standardize_mtx


def top_n_neighbor_by_pcc(std_mtx, K, id_type, id=4321):
    if id_type == 'movie':
        std_mtx = std_mtx.T
    id_vec = std_mtx[id, :]
    # dot product of the standardize matrix gives PCC value
    # NOTE: cos gives the same results
    neighbor_mtx = std_mtx.dot(id_vec.T)
    # post dot product, we get a np matrix object
    id_vector = np.asarray(neighbor_mtx.T)[0]
    # we need to remove the user/movie itself as it would always be the closest
    id_vector_top_K = np.argsort(id_vector, kind='mergesort')[::-1][1:K+1]

    return id_vector_top_K, id_vector[id_vector_top_K]


def pcc_based_prediction(input_pair, csr_mtx, std_sim_mtx, weight=False, K=10):
    # here we use the user-based, i.e. the memory based method but replace
    # similarity metric to PCC
    movie_id, user_id = input_pair
    user_pcc_vec = std_sim_mtx[user_id, :]
    user_knn = np.argsort(user_pcc_vec, kind='mergesort')[::-1][1:K+1]
    user_knn_val = user_pcc_vec[user_knn]
    # still referring back to the original matrix for rating prediction
    knn_movie_rating = csr_mtx[user_knn, :]
    if not weight:
        # for mean average
        avg_rating = knn_movie_rating[:, movie_id].toarray().mean()
    else:
        total_wtg = np.absolute(user_knn_val).sum()
        # avoid zero division
        if total_wtg == 0:
            total_wtg = PCC_DEFAULT_DENOM
        wtg_rating = user_knn_val.dot(knn_movie_rating[:, movie_id].toarray())[0]
        avg_rating = wtg_rating/total_wtg

    return avg_rating + 3.0


def run_memory_based_knn(movie_user_pairlist, csr_mtx, K):
    avg_dot, avg_cos, avg_wtg_dot, avg_wtg_cos = [], [], [], []
    start_time = time.time()
    for pair in movie_user_pairlist:
        avg_dot_r = memory_based_prediction(pair, csr_mtx, top_n_neighbor_by_dot, False, K)
        avg_dot.append(avg_dot_r)
    print("Time for UserBased_DOT_AVG_{}: ".format(K) + "--- %s seconds ---" % (time.time() - start_time))
    write_predict_into_txt(avg_dot, WRITE_DEV_PATH + "UserBased_DOT_AVG_{}.txt".format(K))

    # start_time = time.time()
    # for pair in movie_user_pairlist:
    #     avg_wtg_dot_r = memory_based_prediction(pair, csr_mtx, top_n_neighbor_by_dot, True, K)
    #     avg_wtg_dot.append(avg_wtg_dot_r)
    # print("Time for UserBased_DOT_WTG_{}: ".format(K) + "--- %s seconds ---" % (time.time() - start_time))
    # write_predict_into_txt(avg_wtg_dot, WRITE_DEV_PATH + "UserBased_DOT_WTG_{}.txt".format(K))

    start_time = time.time()
    for pair in movie_user_pairlist:
        avg_cos_r = memory_based_prediction(pair, csr_mtx, top_n_neighbor_by_cos, False, K)
        avg_cos.append(avg_cos_r)
    write_predict_into_txt(avg_cos, WRITE_DEV_PATH+"UserBased_COS_AVG_{}.txt".format(K))
    print("Time for UserBased_COS_AVG_{}: ".format(K) + "--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    for pair in movie_user_pairlist:
        avg_wtg_cos_r = memory_based_prediction(pair, csr_mtx, top_n_neighbor_by_cos, True, K)
        avg_wtg_cos.append(avg_wtg_cos_r)
    write_predict_into_txt(avg_wtg_cos, WRITE_DEV_PATH+"UserBased_COS_WTG_{}.txt".format(K))
    print("Time for UserBased_COS_WTG_{}: ".format(K) + "--- %s seconds ---" % (time.time() - start_time))


def run_item_based_knn(movie_user_pairlist, csr_mtx, K):
    sim_dot_start =time.time()
    item_dot_mtx = csr_mtx.T.dot(csr_mtx).toarray()
    sim_dot_time = time.time()-sim_dot_start
    sim_cos_start =time.time()
    item_cos_mtx = cosine_similarity(csr_mtx.T, csr_mtx.T)
    sim_cos_time = time.time()-sim_cos_start

    avg_dot, avg_cos, avg_wtg_dot, avg_wtg_cos = [], [], [], []
    start_time = time.time()
    for pair in movie_user_pairlist:
        avg_dot_r = item_based_prediction(pair, csr_mtx, item_dot_mtx, False, K)
        avg_dot.append(avg_dot_r)
    print("Time for MovieBased_DOT_AVG_{}: ".format(K) + "--- %s seconds ---" % (time.time() - start_time+sim_dot_time))
    write_predict_into_txt(avg_dot, WRITE_DEV_PATH + "MovieBased_DOT_AVG_{}.txt".format(K))


    # start_time = time.time()
    # for pair in movie_user_pairlist:
    #     avg_wtg_dot_r = item_based_prediction(pair, csr_mtx, item_dot_mtx, True, K)
    #     avg_wtg_dot.append(avg_wtg_dot_r)
    # print("Time for MovieBased_DOT_WTG_{}: ".format(K) + "--- %s seconds ---" % (time.time() - start_time+sim_dot_time))
    # write_predict_into_txt(avg_wtg_dot, WRITE_DEV_PATH + "MovieBased_DOT_WTG_{}.txt".format(K))

    start_time = time.time()
    for pair in movie_user_pairlist:
        avg_cos_r = item_based_prediction(pair, csr_mtx, item_cos_mtx, False, K)
        avg_cos.append(avg_cos_r)
    write_predict_into_txt(avg_cos, WRITE_DEV_PATH+"MovieBased_COS_AVG_{}.txt".format(K))
    print("Time for MovieBased_COS_AVG_{}: ".format(K) + "--- %s seconds ---" % (time.time() - start_time+sim_cos_time))

    start_time = time.time()
    for pair in movie_user_pairlist:
        avg_wtg_cos_r = item_based_prediction(pair, csr_mtx, item_cos_mtx, True, K)
        avg_wtg_cos.append(avg_wtg_cos_r)
    write_predict_into_txt(avg_wtg_cos, WRITE_DEV_PATH+"MovieBased_COS_WTG_{}.txt".format(K))
    print("Time for MovieBased_COS_WTG_{}: ".format(K) + "--- %s seconds ---" % (time.time() - start_time+sim_cos_time))


def run_pcc_based_knn(movie_user_pairlist, csr_mtx, K):
    avg_dot, avg_cos, avg_cos_wtg = [], [], []

    std_start = time.time()
    std_mtx = standardize_mtx(csr_mtx)
    std_time = time.time()-std_start

    std_dot_start = time.time()
    std_dot_mtx = np.asarray(std_mtx.dot(std_mtx.T))
    std_dot_time = time.time()-std_dot_start+std_time

    std_cos_start = time.time()
    std_cos_mtx = cosine_similarity(np.asarray(std_mtx), np.asarray(std_mtx))
    std_cos_time = time.time() - std_cos_start + std_time

    start_time = time.time()
    for pair in movie_user_pairlist:
        avg_dot_r = pcc_based_prediction(pair, csr_mtx, std_dot_mtx, False, K)
        avg_dot.append(avg_dot_r)
    print("Time for PCCBased_DOT_AVG_{}: ".format(K) + "--- %s seconds ---" % (time.time() - start_time+std_dot_time))
    write_predict_into_txt(avg_dot, WRITE_DEV_PATH + "PCCBased_DOT_AVG_{}.txt".format(K))

    start_time = time.time()
    for pair in movie_user_pairlist:
        avg_cos_r = pcc_based_prediction(pair, csr_mtx, std_cos_mtx, False, K)
        avg_cos.append(avg_cos_r)
    print("Time for PCCBased_COS_AVG_{}: ".format(K) + "--- %s seconds ---" % (time.time() - start_time+std_cos_time))
    write_predict_into_txt(avg_cos, WRITE_DEV_PATH + "PCCBased_COS_AVG_{}.txt".format(K))

    start_time = time.time()
    for pair in movie_user_pairlist:
        avg_cos_wtg_r = pcc_based_prediction(pair, csr_mtx, std_cos_mtx, True, K)
        avg_cos_wtg.append(avg_cos_wtg_r)
    print("Time for PCCBased_COS_WTG_{}: ".format(K) + "--- %s seconds ---" % (time.time() - start_time+std_cos_time))
    write_predict_into_txt(avg_cos_wtg, WRITE_DEV_PATH + "PCCBased_COS_WTG_{}.txt".format(K))

