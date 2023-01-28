import CNN
import LSTM
import time
from utils import *


if __name__ == "__main__":
    # first explore data
    explore_data(TRAIN_PATH)

    # start running models
    # read data with regex handling
    train_data_df = read_all_sentiment_data(TRAIN_PATH,regex=True)
    test_data_df = read_all_sentiment_data(TEST_PATH, regex=True)
    # create encoder and pre-process training and testing data
    encoder, train_top_token = create_data_encoder(train_data_df)
    train_data, train_sentiment= data_preprocess(encoder, train_data_df)
    test_data, test_sentiment = data_preprocess(encoder, test_data_df)
    # create embedding matrix using top tokens from training set ONLY
    embed_mtx=create_embed_mtx(train_top_token)

    # for RNN/LSTM, no pretrain > pretrain
    start_time = time.time()
    lstm_fit, lstm_eval = LSTM.run_lstm(train_data, train_sentiment, test_data, test_sentiment)
    print("LSTM no pretrain runtime: {}".format(time.time() - start_time))

    start_time = time.time()
    pretrain_lstm_fit, pretrain_lstm_eval = LSTM.run_lstm(
        train_data, train_sentiment, test_data, test_sentiment, embed_mtx)
    print("LSTM pretrain runtime: {}".format(time.time() - start_time))

    # for CNN, no pretrain > pretrain
    start_time = time.time()
    cnn_fit, cnn_eval = CNN.run_cnn(train_data, train_sentiment, test_data, test_sentiment)
    print("CNN no pretrain runtime: {}".format(time.time() - start_time))

    start_time = time.time()
    pretrain_cnn_fit, pretrain_cnn_eval = CNN.run_cnn(
        train_data, train_sentiment, test_data, test_sentiment, embed_mtx)
    print("CNN pretrain runtime: {}".format(time.time() - start_time))

    # print out the plots
    plot_model_result(lstm_fit, "LSTM", "Accuracy")
    plot_model_result(lstm_fit, "LSTM", "Loss")

    plot_model_result(pretrain_lstm_fit, "Pretrain LSTM", "Accuracy")
    plot_model_result(pretrain_lstm_fit, "Pretrain LSTM", "Loss")

    plot_model_result(cnn_fit, "CNN", "Accuracy")
    plot_model_result(cnn_fit, "CNN", "Loss")

    plot_model_result(pretrain_cnn_fit, "Pretrain CNN", "Accuracy")
    plot_model_result(pretrain_cnn_fit, "Pretrain CNN", "Loss")