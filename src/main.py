import sys
from sklearn.datasets import load_svmlight_file
from scipy.sparse import hstack
from svm import *
import pandas as pd

PARA_PATH = 'resources/optval.txt'

def create_para_table():
    para_df = pd.DataFrame(
        data={'dataset':['covtype', 'realsim'],
              'lamb': [3631.32, 7230.87],
              'optval':[2541.665, 669.665],
              'learningR': [0.00003, 0.0003],
              'iteration': [50, 50],
              'beta': [0.1, 0.01],
              'batch': [1000, 500]},
    ).set_index('dataset')

    return para_df


def main():
    #train_file = "data/covtype.scale.trn.libsvm"
    #test_file = "data/covtype.scale.tst.libsvm"
    #train_file_r = "data/realsim.scale.trn.libsvm"
    #test_file_r = "data/realsim.scale.tst.libsvm"
    parameter_table = create_para_table()
    # use load_svmlight_file to load data from train_file
    train_file = sys.argv[1]
    train_X, train_y = load_svmlight_file(train_file)

    # use load_svmlight_file to load data from test_file
    test_file = sys.argv[2]
    test_X, test_y = load_svmlight_file(test_file)

    # adding one extra column for bias terms
    train_X = hstack((train_X, np.ones(train_X.shape[0]).reshape((train_X.shape[0], 1)))).tocsr()
    test_X = hstack((test_X, np.ones(test_X.shape[0]).reshape((test_X.shape[0], 1)))).tocsr()

    # get the dataset
    dataset = train_file.split("/")[-1].split(".")[0]

    # run SVM by creating SVM object
    dimension = train_X.shape[1]
    svm_sdg = SVM(dimension, dataset, parameter_table)
    svm_newton = SVM(dimension, dataset, parameter_table)

    # calling Mini-Batch SDG function
    sdg_results = svm_sdg.mini_batch_SDG(train_X, train_y, test_X, test_y)

    # calling Newton function
    newton_results = svm_newton.newtonSDG(train_X, train_y, test_X, test_y)

    # plotting results
    for val in FIG_TITLE_MAP.keys():
        plot_svm_result(sdg_results, newton_results, val, dataset)
        #plot_sdg_result(sdg_results, val, dataset)
        #plot_newton_result(newton_results, val, dataset)

# Main entry point to the program
if __name__ == '__main__':
    main()
