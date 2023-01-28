from utils import *
import keras as kr

EMBED_DROPOUT = 0.2
LSTM_DROPOUT = 0.4
LSTM_BATCH = 64
LSTM_EPOCH = 5  # 150


def model_lstm(embed_mtx=None):
    # initialize input layer with embedding size
    input = kr.layers.Input(shape=(EMBED_VEC_SIZE,))

    # Layer 1: create word2vec embedding, note for LSTM, we trained the embedding matrix if there is no pre-trained data
    # otherwise we don't train with the input pre-trained embedding
    trainable = True
    if embed_mtx is not None:
        embed_mtx = [embed_mtx]
        trainable = False
    embedding = kr.layers.Embedding(TOP_WORD_INDEX, EMBED_VEC_SIZE, trainable=trainable, weights=embed_mtx)(input)

    # Layer 2: SpatialDropout1D() - for embedding, we dropout certain percentage to avoid overfitting
    embedding = kr.layers.SpatialDropout1D(EMBED_DROPOUT)(embedding)

    # Layer 3: lstm layer - add dropout rate to avoid overfitting
    lstm = kr.layers.LSTM(EMBED_VEC_SIZE, dropout=LSTM_DROPOUT,
                          recurrent_dropout=LSTM_DROPOUT, activation="tanh")(embedding)
    # Layer X (drop): linear layer - we use linear activation to fully connected the lstm layers
    # lstm = kr.layers.Dense(EMBED_VEC_SIZE, activation="linear")(lstm)

    # Layer 4: Output layer - use sigmoid activation to reach output layer with 1 value for sentiment
    # softmax under-perform sigmoid hence we use sigmoid instead
    output = kr.layers.Dense(1, activation="sigmoid")(lstm)

    # create model using above layers and compile with optimizers
    model = kr.models.Model(inputs=input, outputs=output)
    adam = kr.optimizers.Adam()
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    # print out model summary for checking purpose
    model.summary()

    return model


def run_lstm(train_vec, train_stmt, test_vec, test_stmt, embed_mtx=None):
    # run LSTM with model fitting and evaluation
    lstm_model = model_lstm(embed_mtx)
    # fit model with train and test data, set batch size and epochs accordingly
    fit_model = lstm_model.fit(train_vec, train_stmt,
                               batch_size=LSTM_BATCH, epochs=LSTM_EPOCH, validation_data=(test_vec, test_stmt))
    evals = lstm_model.evaluate(test_vec, test_stmt, batch_size=LSTM_BATCH)

    return fit_model, evals