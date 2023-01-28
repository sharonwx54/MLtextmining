import numpy as np
import conjugateGradient as cg
import matplotlib.pyplot as plt
import time

FIG_TITLE_MAP = {'rv': "Relative Func Value vs Training Time",
                 'norm': "Gradient Norm vs Training Time",
                 'acc': "Accuracy vs Training Time"}
FIG_LIST_MAP = {'rv': 0, 'norm': 2, 'acc': 1}


class SVM(object):
    """SVM

    :param object:
    """
    def __init__(self, dimension, dataset, para_table):
        # initialize all parameters based on input table and dataset name
        self.wt_mtx = np.random.randn(dimension)
        paras = para_table[para_table.index == dataset]
        self.optval = paras.optval.values[0]
        self.lamb = paras.lamb.values[0]
        self.beta = paras.beta.values[0]
        self.learningR = paras.learningR.values[0]
        self.iteration = paras.iteration.values[0]
        self.batch = paras.batch.values[0]

    def mini_batch_SDG(self, train_x, train_y, test_x, test_y):
        print("Starting Mini-Batch SDG")
        datanum = train_x.shape[0]
        rv_loss_list, acc_list, gd_norm_list, time_list = [], [], [], []
        start = time.time()
        for i in range(self.iteration):
            print("Entering iteration {}".format(i))
            batch_x, batch_y = None, None
            curr_rate = self.learningR / (1 + self.beta * i)
            # forming mini-batch
            for m in range(0, datanum, self.batch):
                # take the slice value of each batch
                batch_x = train_x[m:m+self.batch, :]
                batch_y = train_y[m:m+self.batch]
                # calculation the non-regularization part of losses
                # and take only non-zero x_i entries
                svm_vals = 1 - batch_y*batch_x.dot(self.wt_mtx)
                pos_index = np.squeeze((svm_vals > 0)) # set I
                x_batch_pos = batch_x[pos_index, :]

                # calculate gradient by (5) in the handout
                grad_xw_y = x_batch_pos.dot(self.wt_mtx) - batch_y[pos_index]
                grad_wt_mtx = self.wt_mtx + (2*self.lamb/self.batch)*x_batch_pos.T.dot(grad_xw_y)

                gd_norm = np.linalg.norm(grad_wt_mtx, 2)
                gd_norm_list.append(gd_norm)

                # update weight matrix by gradient
                self.wt_mtx = self.wt_mtx - curr_rate * grad_wt_mtx
                time_list.append(time.time() - start)

                # calculate loss
                loss_no_reg = (self.lamb/self.batch)*np.dot(np.maximum(svm_vals, 0).T, np.maximum(svm_vals, 0))
                loss = 0.5*np.dot(self.wt_mtx.T, self.wt_mtx) + loss_no_reg
                rv_loss = (loss - self.optval)/self.optval
                accuracy = self.calc_accuracy(test_x, test_y)

                rv_loss_list.append(rv_loss)
                acc_list.append(accuracy)

        print("Total Training Time for Mini-Batch SDG", time_list[-1])
        print("Final RV for Mini-Batch SDG:", rv_loss_list[-1])
        print("Accuracy for Mini-Batch SDG:", acc_list[-1])
        print("Gradient Norm for Mini-Batch SDG:", gd_norm_list[-1])
        print("Mini-Batch SDG DONE")
        return rv_loss_list, acc_list, gd_norm_list, time_list

    def newtonSDG(self,  train_x, train_y, test_x, test_y):
        print("Starting Newton Method")
        datanum = train_x.shape[0]
        rv_loss_list, acc_list, gd_norm_list, time_list = [], [], [], []
        start = time.time()

        # for newton, 1/5 of original iteration is good enough
        newton_iteration = (self.iteration/5).astype(int)
        print("Total Newton Iteration is "+str(newton_iteration))
        for i in range(newton_iteration):
            print("Entering iteration {}".format(i))
            svm_vals = 1- train_y*train_x.dot(self.wt_mtx)
            # take only non-zero x_i entries
            pos_index = np.squeeze((svm_vals > 0))  # set I
            train_x_pos = train_x[pos_index, :]

            # calculate gradient by (5) in the handout
            grad_xw_y = train_x_pos.dot(self.wt_mtx) - train_y[pos_index]
            grad_wt_mtx = self.wt_mtx + (2*self.lamb/datanum)*train_x_pos.T.dot(grad_xw_y)

            gd_norm = np.linalg.norm(grad_wt_mtx, 2)
            gd_norm_list.append(gd_norm)

            # calculate hessian using given function
            grad_over_hessian, iter_num = cg.conjugateGradient(train_x, pos_index, grad_wt_mtx, self.lamb)
            # update weight matrix by gradient
            self.wt_mtx = self.wt_mtx + grad_over_hessian

            # calculate loss
            loss_no_reg = (self.lamb / datanum) * np.dot(np.maximum(svm_vals, 0).T, np.maximum(svm_vals, 0))
            loss = 0.5 * np.dot(self.wt_mtx.T, self.wt_mtx) + loss_no_reg
            rv_loss = (loss - self.optval) / self.optval
            accuracy = self.calc_accuracy(test_x, test_y)
            #  appending values
            time_list.append(time.time() - start)
            rv_loss_list.append(rv_loss)
            acc_list.append(accuracy)

        print("Total Training Time for Newton", time_list[-1])
        print("Final RV for Newton:", rv_loss_list[-1])
        print("Accuracy for Newton:", acc_list[-1])
        print("Gradient Norm for Newton:", gd_norm_list[-1])
        print("Newton Method DONE")
        return rv_loss_list, acc_list, gd_norm_list, time_list

    def calc_accuracy(self, test_x, test_y):
        # first predict the pred_y based on test_x
        pred_y = np.zeros(test_x.shape[0])
        svm_vals = test_x.dot(self.wt_mtx)
        for i in range(test_x.shape[0]):
            if svm_vals[i] >= 0:
                pred_y[i] = 1
            else:
                pred_y[i] = -1
        # at the end calculate accuracy
        accuracy = 100*np.mean(test_y == pred_y)

        return accuracy


def plot_svm_result(sdg_results, newton_results, fig_title, dataset_name):
    plt.figure(fig_title)
    plt.title(FIG_TITLE_MAP[fig_title]+" "+dataset_name)
    plt.plot(sdg_results[3], sdg_results[FIG_LIST_MAP[fig_title]], "skyblue", linewidth=1, label='Mini-Batch SDG')
    plt.plot(newton_results[3], newton_results[FIG_LIST_MAP[fig_title]], 'purple', linewidth=2, label='Newton')
    plt.legend(loc='upper right')


def plot_sdg_result(sdg_results, fig_title, dataset_name):
    plt.figure(fig_title+"Mini-Batch")
    plt.title(FIG_TITLE_MAP[fig_title]+" Mini-Batch SDG "+ dataset_name)
    plt.plot(sdg_results[3], sdg_results[FIG_LIST_MAP[fig_title]], "skyblue", linewidth=1)


def plot_newton_result(newton_results, fig_title, dataset_name):
    plt.figure(fig_title+"Newton")
    plt.title(FIG_TITLE_MAP[fig_title]+" Newton "+ dataset_name)
    plt.plot(newton_results[3], newton_results[FIG_LIST_MAP[fig_title]], 'purple', linewidth=1)
