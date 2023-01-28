import data_preprocess as dp
import feature_selection as fs
from RMLR import *
import os
from liblinear.liblinearutil import *

TEST_PRED_PATH = "test-predictions.txt"
DEV_PRED_PATH = "dev-predictions.txt"
RMLR_LR = 0.001
RMLR_LAMBDA = 0.002


def rmlr_by_feature(star_mtx, feature_type):
    # training rmlr by different features
    with open('{}_feature_train.pickle'.format(feature_type), 'rb') as f_train:
        train_f = pickle.load(f_train)

    # split data into 80-20 train vs validation
    # since the data itself is quite random, simply taking the first 80 vs last 20 is fair
    train_threshold = int(0.8*train_f.shape[0])
    train_x = train_f[:train_threshold, :]
    valid_x = train_f[train_threshold:, :]
    train_y = star_mtx[:train_threshold, :]
    valid_y = star_mtx[train_threshold:, :]

    # now run the rmlr to train
    rmlr = RMLR(RMLR_LR, RMLR_LAMBDA, n_iters=100, batch=1000)
    accs, rmses, losses = rmlr.train_fit(train_x, train_y, valid_x, valid_y)
    print("Accuracy changes as : ")
    print(accs)
    print("RMSE changes as : ")
    print(rmses)
    print("Loss changes as : ")
    print(losses)

    return rmlr


def predict_and_save_rmlr(rmlr, feature_type):
    # now run prediction on dev and test set
    print("Predicting RMLR result for DEV and TEST")
    with open('{}_feature_test.pickle'.format(feature_type), 'rb') as f_test:
        test_f = pickle.load(f_test)
    with open('{}_feature_dev.pickle'.format(feature_type), 'rb') as f_dev:
        dev_f = pickle.load(f_dev)

    # need to add one back as the prediction is the index of feature, which is 1 off
    dev_pred_hard = rmlr.predict_hard(dev_f)+1
    dev_pred_soft = rmlr.predict_soft(dev_f)+1

    test_pred_hard = rmlr.predict_hard(test_f)+1
    test_pred_soft = rmlr.predict_soft(test_f)+1
    print("Writing RMLR predition for DEV and TEST")
    write_to_txt(dev_pred_hard, dev_pred_soft, DEV_PRED_PATH, feature_type.lower())
    write_to_txt(test_pred_hard, test_pred_soft, TEST_PRED_PATH, feature_type.lower())


def rmlr_predict_by_feature(rmlr, train_dic, feature_type):
    # now run prediction on entire train set
    with open('{}_feature_train.pickle'.format(feature_type), 'rb') as f_train:
        train_f = pickle.load(f_train)
    # need to subtract one due to indexing
    train_y = np.asarray(train_dic['stars'])-1
    # calculate accuracy on entire training set, rather than validation set only
    train_acc = rmlr.accuracy(train_f, train_y)
    train_rsme = rmlr.rmse(train_f, train_y)
    print("Metrics for Feature "+feature_type+" >>>>")
    print("Accuracy for TRAIN set is "+str(train_acc))
    print("RSME for train_acc set is "+str(train_rsme))


def svm_main():
    # wrapper function to run svm at once
    print("Preprocessing CTF for SVM")
    svm_preprocessing("CTF", "dev")
    svm_preprocessing("CTF", "test")
    svm_train_preprocessing("CTF")

    print("Preprocessing DF for SVM")
    svm_preprocessing("DF", "test")
    svm_preprocessing("DF", "dev")
    svm_train_preprocessing("DF")

    # for CTF
    y, x = svm_read_problem('SVM_CTF_train.txt')
    ctf_m = train(y, x)
    ctf_labels, ctf_acc, ctf_vals = predict(y, x, ctf_m)
    y_dev, x_dev = svm_read_problem('SVM_CTF_dev.txt')
    ctf_dev_labels, ctf_dev_acc, ctf_dev_vals = predict(y_dev, x_dev, ctf_m)
    writer = open("SVM_CTF_devpred.txt", "w")
    for p in ctf_dev_labels:
        writer.write(str(int(p))+" "+str(int(p)))
        writer.write("\n")
    writer.close()

    # for DF
    y_df, x_df = svm_read_problem('SVM_DF_train.txt')
    df_m = train(y_df, x_df)
    df_labels, df_acc, df_vals = predict(y_df, x_df, df_m)
    y_df_dev, x_df_dev = svm_read_problem('SVM_DF_dev.txt')
    df_dev_labels, df_dev_acc, df_dev_vals = predict(y_df_dev, x_df_dev, df_m)
    writer_dev = open("SVM_DF_devpred.txt", "w")
    for p in df_dev_labels:
        writer_dev.write(str(int(p))+" "+str(int(p)))
        writer_dev.write("\n")
    writer_dev.close()


def rmlr_main(run_all=False):
    # aggregate function for running rmlr at once
    with open('train_data.pickle', 'rb') as data_train:
        train_dic = pickle.load(data_train)
        star_mtx = dp.stars_to_mtx(train_dic['stars'])

    print("Running RMLR with feature engineering")
    eng_rmlr = rmlr_by_feature(star_mtx, "ENG")

    print("Predicting using ENG feature")
    rmlr_predict_by_feature(eng_rmlr, train_dic, "ENG")

    print("Predicting on dev and test using ENG feature")
    predict_and_save_rmlr(eng_rmlr, "ENG")

    if run_all:
        print("Running CTF")
        ctf_rmlr = rmlr_by_feature(star_mtx, "CTF")
        print("Running DF")
        df_rmlr = rmlr_by_feature(star_mtx, "DF")
        # print("Running TFIDF")
        # tfidf_rmlr = rmlr_by_feature(star_mtx, "TFIDF")

        print("Predicting using CTF feature")
        rmlr_predict_by_feature(ctf_rmlr, train_dic, "CTF")
        print("Predicting using DF feature")
        rmlr_predict_by_feature(df_rmlr, train_dic, "DF")
        # print("Predicting using TFIDF feature")
        # rmlr_predict_by_feature(tfidf_rmlr, train_dic, "TFIDF")

        print("Predicting on dev and test using CTF feature")
        predict_and_save_rmlr(ctf_rmlr, "CTF")
        print("Predicting on dev and test using DF feature")
        predict_and_save_rmlr(df_rmlr, "DF")


def preprocess_main():
    # only start entire preprocessing if pickle files do not exists
    if not os.path.isfile("train_data.pickle"):
        print("Starting new preprocessing for raw data file")
        dp.run_preprocessing()
        print("Starting new feature creation for processed file, saving under pickle")
        fs.run_feature_creation()
    else:
        print("Preprocess already done. No need to rerun.")


def main(run_type):
    # function to run different model given command line input
    if run_type == 'svm':
        svm_main()
    elif run_type == "":
        rmlr_main()
    else:
        rmlr_main(True)
