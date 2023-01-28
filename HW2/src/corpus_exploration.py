from src.utils import *
from src.collaborative_filtering import top_n_neighbor_by_dot, top_n_neighbor_by_cos

TOP_NN_FOR_EXPLORATION = 5

def print_basic_stats(corpus_info):
    # print out the result for corpus exploration
    print("the total number of movies: {}".format(len(corpus_info.mov_set)))
    print("the total number of users: {}".format(len(corpus_info.user_set)))
    # print our number by ratings
    print("the number of times any movie was rated 1: {}".format((corpus_info.data == 1).sum()))
    print("the number of times any movie was rated 3: {}".format((corpus_info.data == 3).sum()))
    print("the number of times any movie was rated 5: {}".format((corpus_info.data == 5).sum()))
    # we need to add back 3 to display the proper original rating scale
    print("the average movie rating across all users and movies: {}".format(
        corpus_info.data.mean()))

def print_user_stats(corpus_mtx, user_id):
    # convert coo to csr (corpus_info.X)
    user_based_mtx = corpus_mtx.tocsr()
    user_rating = user_based_mtx[user_id, :]
    print("For user {}:".format(user_id))
    print("the number of movies rated: {}".format(user_rating.getnnz()))
    print("the number of times the user gave a '1' rating: {}".format((user_rating==1).sum()))
    print("the number of times the user gave a '3' rating: {}".format((user_rating==3).sum()))
    print("the number of times the user gave a '5' rating: {}".format((user_rating==5).sum()))
    print("the average movie rating for this user: {}".format(
        user_rating.sum()/user_rating.getnnz()))


def print_movie_stats(corpus_mtx, movie_id):
    # convert coo to csc (corpus_info.X)
    movie_based_mtx = corpus_mtx.tocsr()
    movie_rating = movie_based_mtx[:, movie_id]
    print("For movie {}:".format(movie_id))
    print("the number of users rating this movie: {}".format(movie_rating.getnnz()))
    print("the number of times the user gave a '1' rating: {}".format((movie_rating==1).sum()))
    print("the number of times the user gave a '3' rating: {}".format((movie_rating==3).sum()))
    print("the number of times the user gave a '5' rating: {}".format((movie_rating==5).sum()))
    print("the average rating for this movie: {}".format(
        movie_rating.sum()/movie_rating.getnnz()))


def print_corpus_exploration():
    # reading data from the training set
    corpus_info_raw = load_review_data_matrix(CORPUS_PATH, 0)
    print_basic_stats(corpus_info_raw)
    print_user_stats(corpus_info_raw.X, 4321)
    print_movie_stats(corpus_info_raw.X, 3)


def print_top_NN(coo_mtx, user_id, movie_id):
    user_dot_top, udt_val = top_n_neighbor_by_dot(coo_mtx, TOP_NN_FOR_EXPLORATION, "user", user_id)
    user_cos_top, uct_val = top_n_neighbor_by_cos(coo_mtx, TOP_NN_FOR_EXPLORATION, "user", user_id)
    movie_dot_top, idt_val = top_n_neighbor_by_dot(coo_mtx, TOP_NN_FOR_EXPLORATION, "movie", movie_id)
    movie_cos_top, ict_val = top_n_neighbor_by_cos(coo_mtx, TOP_NN_FOR_EXPLORATION, "movie", movie_id)
    print("Top 5 NNs of user {} in terms of dot product similarity: {}".format(user_id, user_dot_top))
    print("Top 5 NNs of user {} in terms of cosine similarity: {}".format(user_id, user_cos_top))
    print("Top 5 NNs of movie {} in terms of dot product similarity: {}".format(movie_id, movie_dot_top))
    print("Top 5 NNs of movie {} in terms of cosine similarity: {}".format(movie_id, movie_cos_top))