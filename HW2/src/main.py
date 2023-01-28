from src.corpus_exploration import *
from src.prob_mtx_factorization import *
from src.collaborative_filtering import *


if __name__ == '__main__':
    dev_pairs = load_dev_test_data(DEV_PATH)
    test_pairs = load_dev_test_data(TEST_PATH)
    corpus_info = load_review_data_matrix(CORPUS_PATH, 3)
    coo_mtx = corpus_info.X
    csr_mtx = coo_mtx.tocsr()

    # For question 1
    print_corpus_exploration()
    print_top_NN(coo_mtx, 4321, 3)

    # For question 2 - 3
    for K in [10, 100, 500]:
        print("User-Based KNN - {}".format(K))
        run_memory_based_knn(dev_pairs, csr_mtx, K)
        print("Item-Based KNN - {}".format(K))
        run_item_based_knn(dev_pairs, csr_mtx, K)
        print("PCC-User-Based KNN - {}".format(K))
        run_pcc_based_knn(dev_pairs, csr_mtx, K)

    # For question 4
    print("PMF on DEV")
    run_pmf_prediction(dev_pairs, latents=LATENTS, test_file=None)
    print("PMF on TEST")
    run_pmf_prediction(test_pairs, latents=[10], test_file="test-prediction.txt")