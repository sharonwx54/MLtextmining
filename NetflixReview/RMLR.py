from data_preprocess import *
import numpy as np

class RMLR:
    def __init__(self, learning_rate=0.1, lam=0, n_iters=500, batch=1000):
        self.lr = learning_rate
        self.lam = lam
        self.iters = n_iters
        self.batch = batch
        # initialize weight matrix with random numbers
        self.W = np.random.randn(TOP_K_FOR_FEATURE, NUM_FEATURE)

    def train_fit(self, train_x, train_y, test_x, test_y):
        train_num = train_x.shape[0]
        # data is actually quite sparse, so need to filter out zero-value empty data
        test_y_pos = np.nonzero(test_y)[1]
        acc_list, rmse_list, loss_list = [], [], []
        for iter in range(self.iters):
            batch_x, batch_y = None, None
            # for each batch
            for m in range(0, train_num, self.batch):
                batch_x = train_x[m:m+self.batch, :]
                batch_y = train_y[m:m+self.batch]
                nom = np.exp(batch_x.dot(self.W))
                denom = np.sum(nom, axis=1)
                prob = nom/np.reshape(denom, (-1, 1))
                prob_sum = np.sum(prob*batch_y, axis=1)
                loss = np.sum(np.log(prob_sum))-0.5*self.lam*np.linalg.norm(self.W)**2

                # calculate the derivative per star value
                dW_list = []
                for i in range(NUM_FEATURE):
                    inner_val = np.reshape(batch_y[:, i]-(nom[:, i]/denom), (-1, 1))
                    dW = np.array(batch_x.multiply(inner_val).sum(axis=0))[0:].reshape(-1, 1)
                    dW_list.append(dW)
                dW_agg = np.concatenate(dW_list, axis=1) - self.lam*self.W
                self.W += self.lr*dW_agg

            if (iter%25==0):
                accuracy = self.accuracy(test_x, test_y_pos)
                rmse = self.rmse(test_x, test_y_pos)
                acc_list.append(accuracy) # last one is the final accuracy on validation set
                rmse_list.append(rmse) # last one is the final rsme on validation set
                loss_list.append(loss) # last one is the final loss on validation set

        return acc_list, rmse_list, loss_list

    def predict_soft(self, test_x):
        nom = np.exp(test_x.dot(self.W))
        denom = np.sum(nom, axis=1)
        pred_y = 0
        for star in range(0, 5):
            pred_y += star*(np.exp(test_x.dot(self.W[:, star]))/denom)

        return pred_y

    def predict_hard(self, test_x):
        nom = np.exp(test_x.dot(self.W))
        denom = np.sum(nom, axis=1)
        prob = nom/np.reshape(denom, (-1, 1))
        pred_y = np.argmax(prob, axis=1)

        return pred_y

    def rmse(self, test_x, test_y):
        pred_y = self.predict_soft(test_x)
        rmse = np.sqrt(np.sum((test_y-pred_y)*(test_y-pred_y), axis=0)/test_y.shape[0])

        return rmse

    def accuracy(self, test_x, test_y):
        pred_y = self.predict_hard(test_x)
        accuracy = np.mean(test_y == pred_y)

        return accuracy


def write_to_txt(hard, soft, path, feature):
    # write result to txt
    assert(len(soft) == len(hard))
    if feature == "eng":
        writer = open(path, 'w')
    else:
        writer = open(feature+"-"+path, 'w')
    for y in range(len(soft)):
        writer.write(str(hard[y])+" "+str(soft[y]))
        writer.write("\n")
    writer.close()
