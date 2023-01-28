from src.utils import *
import torch as tr
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.autograd import Variable
import time


LATENTS = [2, 5, 10, 20, 50]
LOSS_ITERATION = 1000
# try rate at 0.005 given by the paper
LEARNING_RATE = 0.005
SIGMA_MULTIPLIER = 4


def create_ind_mtx(corpus_info):
    # create indicator function for user-movie matrix
    # dimension is user_num x mov_num
    ind_mtx = csr_matrix(
        (np.ones(corpus_info.data.shape[0]), (corpus_info.rows, corpus_info.cols))).todense()

    return ind_mtx


def pmf_by_latent(adj_csr_mtx, ind_mtx, latent):
    # get total number for forming U and V
    user_num, mov_num = adj_csr_mtx.shape
    R = Variable(tr.from_numpy(np.array(adj_csr_mtx)).type('torch.FloatTensor'))
    I = Variable(tr.from_numpy(np.array(ind_mtx)).type('torch.FloatTensor'))
    # create latent U and V using torch normal function - random numbers drawn
    # each has mean zero and std at 1/latent for initialization
    U = Variable(
            tr.normal(tr.zeros((latent, user_num)), tr.ones((latent, user_num)) / (latent)),
            requires_grad=True)
    V = Variable(
            tr.normal(tr.zeros((latent, mov_num)), tr.ones((latent,  mov_num)) / (latent)),
            requires_grad=True)
    U = Parameter(U)
    V = Parameter(V)

    # optimizer = optim.SGD([U, V], lr=0.005, momentum=0.9)
    # Use Adam rather than SGD as it performs faster
    # convert into parameter objects for optimizing purpose
    optimizer = optim.Adam([U, V], lr=LEARNING_RATE)
    # set the sigma to be 4 times initial std of Ind matrix
    sigma = SIGMA_MULTIPLIER*tr.std(I).detach()
    # iterate through the loss iteration, after experiment at 1000 is good enough
    for i in range(LOSS_ITERATION):
        # updating lambda U and V dynamically
        l_U = sigma/tr.std(U).detach()
        l_V = sigma/tr.std(V).detach()
        # l_U = 0.01
        # l_V = 0.001

        # calculate three part of losses
        loss_a = 0.5*(I*tr.pow((R-tr.matmul(U.T, V)), 2)).sum()
        loss_b = 0.5*l_U*tr.pow(tr.norm(U), 2)
        loss_c = 0.5*l_V*tr.pow(tr.norm(V), 2)
        loss = loss_a+loss_b+loss_c
        # sets the gradients of all optimized tensors
        optimizer.zero_grad()
        # take loss backward
        loss.backward()
        # step the optimizer for the next iteration
        optimizer.step()

    # tr.save([U, V], "latent-{}.pt".format(latent))
    pre_mtx = tr.matmul(U.T, V).data.numpy()
    pre_mtx = np.clip(pre_mtx*4+1, 1, 5)

    return pre_mtx


def pmf_prediction_by_latent(to_prd_pairs, predict_mtx, latent, spec_txt=None):
    prd_list = []
    for pair in to_prd_pairs:
        movie_id, user_id = pair
        prd_list.append(predict_mtx[user_id, movie_id])
    if spec_txt:
        write_predict_into_txt(prd_list, WRITE_DEV_PATH + spec_txt)
    else:
        write_predict_into_txt(prd_list, WRITE_DEV_PATH + "PMF_LATENT{}.txt".format(latent))


# run_pmf_prediction(dev_pairs, latents=[2, 5, 20, 50], test_file=None)
def run_pmf_prediction(prd_pairs, latents=LATENTS, test_file=None):
    # we should use the unormalized one here based on the paper
    init_start = time.time()
    corpus_info = load_review_data_matrix(CORPUS_PATH, 0)
    csr_mtx_unnorm = corpus_info.X.tocsr()
    # according to the paper, adjust the value by subtracting one and dividing by four
    # use this method to scale the rating into 0-1
    adj_csr_mtx = (csr_mtx_unnorm.todense() - 1) / 4
    # create indicator matrix
    ind_mtx = create_ind_mtx(corpus_info)
    init_time = time.time()-init_start
    for latent in latents:
        start_time = time.time()
        predict_mtx = pmf_by_latent(adj_csr_mtx, ind_mtx, latent)
        pmf_prediction_by_latent(prd_pairs, predict_mtx, latent, spec_txt=test_file)
        print("Time for PMF_{}: ".format(latent) + "--- %s seconds ---" % (time.time() - start_time+init_time))

