from utils import *
import keras as kr

EMBED_DROPOUT = 0.2
CNN_DROPOUT = 0.5
CNN_FILTER_NUM = 100
CNN_FILTER_LEN = 3

CNN_EPOCH = 10
CNN_BATCH = 32


def model_cnn(embed_mtx=None):
    # initialize input layer with embedding size
    input = kr.layers.Input(shape=(EMBED_VEC_SIZE,))

    # Add the word embedding Layer
    # Layer 1: create word2vec embedding, default to None
    # NOTE for CNN, we don't train the embedding matrix regardless. CNN is efficient enough in learning
    # that even without pre-trained embedding, it converges quickly without updating empty embedding matrix
    if embed_mtx is not None:
        embed_mtx = [embed_mtx]
    embedding = kr.layers.Embedding(TOP_WORD_INDEX, EMBED_VEC_SIZE, trainable=False, weights=embed_mtx)(input)

    # Layer 2: SpatialDropout1D() - for embedding, we dropout certain percentage to avoid overfitting
    embedding = kr.layers.SpatialDropout1D(EMBED_DROPOUT)(embedding)

    # Layer 3: CNN with 100 filters each with length 3, using relu activation
    cnn = kr.layers.Conv1D(CNN_FILTER_NUM, CNN_FILTER_LEN, activation="relu")(embedding)

    # Layer 4: Max Pooling - directly use global as we don't introduce inner layer here
    pooling = kr.layers.GlobalMaxPool1D()(cnn)

    # Layer 5: Fully connected layer with relu activation on pooling layers
    full = kr.layers.Dense(CNN_FILTER_NUM, activation="relu")(pooling)
    # here we dropout 50% to avoid overfitting
    full = kr.layers.Dropout(CNN_DROPOUT)(full)

    # Layer 6: Output using sigmoid - again, sigmoid outperform softmax here
    output = kr.layers.Dense(1, activation="sigmoid")(full)

    # create model using above layers and compile with optimizers
    cnn_model = kr.models.Model(inputs=input, outputs=output)
    adam = kr.optimizers.Adam()
    cnn_model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    # print out model summary for checking purpose
    cnn_model.summary()
    return cnn_model


def run_cnn(train_vec, train_stmt, test_vec, test_stmt, embed_mtx=None):
    # run CNN with model fitting and evaluation
    cnn_model = model_cnn(embed_mtx)
    # fit model with train and test data, set batch size and epochs accordingly
    fit_model = cnn_model.fit(train_vec, train_stmt,
                              batch_size=CNN_BATCH, epochs=CNN_EPOCH, validation_data=(test_vec, test_stmt))
    evals = cnn_model.evaluate(test_vec, test_stmt, batch_size=CNN_BATCH)

    return fit_model, evals