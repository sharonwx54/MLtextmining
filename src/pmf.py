import numpy as np


class PMF(object):
    """PMF

    :param object:
    """

    def __init__(self, num_factors, num_users, num_movies):
        """__init__

        :param num_factors:
        :param num_users:
        :param num_movies:
        """
        # note that you should not modify this function
        np.random.seed(11)
        self.U = np.random.normal(size=(num_factors, num_users))
        self.V = np.random.normal(size=(num_factors, num_movies))
        self.num_users = num_users
        self.num_movies = num_movies

    def predict(self, user, movie):
        """predict

        :param user:
        :param movie:
        """
        # note that you should not modify this function
        return (self.U[:, user] * self.V[:, movie]).sum()

    def train(self, users, movies, ratings, alpha, lambda_u, lambda_v,
              batch_size, num_iterations):
        """train

        :param users: np.array of shape [N], type = np.int64
        :param movies: np.array of shape [N], type = np.int64
        :param ratings: np.array of shape [N], type = np.float32
        :param alpha: learning rate
        :param lambda_u:
        :param lambda_v:
        :param batch_size:
        :param num_iterations: how many SGD iterations to run
        """
        # modify this function to implement mini-batch SGD
        # for the i-th training instance,
        # user `users[i]` rates the movie `movies[i]`
        # with a rating `ratings[i]`.

        total_training_cases = users.shape[0]
        for i in range(num_iterations):
            start_idx = (i * batch_size) % total_training_cases
            users_batch = users[start_idx:start_idx + batch_size]
            movies_batch = movies[start_idx:start_idx + batch_size]
            ratings_batch = ratings[start_idx:start_idx + batch_size]
            curr_size = ratings_batch.shape[0]

            # Calculate the SGD loss on batches and update U and V respectively
            self.calc_SGD_E(users, movies, ratings, alpha, lambda_u, lambda_v)

    def calc_SGD_E(self, users, movies, ratings, alpha, lambda_u, lambda_v):
        """
        Function to calculate the SDG for the E/Loss defined in PMF paper
        :param users: np.array of shape [N], type = np.int64
        :param movies: np.array of shape [N], type = np.int64
        :param ratings: np.array of shape [N], type = np.float32
        :param alpha: learning rate
        :param lambda_u:
        :param lambda_v:
        :return:
        """
        unique_user = set(users)
        unique_movie = set(movies)
        U_dot_V = np.dot(self.U.T, self.V)
        UV_subset = U_dot_V[users, movies]
        # take the gradient on U
        for uid in unique_user:
            # simply take the first one if repeated
            uindex = np.where(users==uid)[0]
            mid = movies[uindex]
            U_UVterm = np.dot(self.V[:, mid], (UV_subset[uindex] - ratings[uindex]))
            Uterm = lambda_u*self.U[:, uid]
            U_SDG = U_UVterm + Uterm
            # update the U by the gradient
            self.U[:, uid] = self.U[:, uid] - alpha*U_SDG

        # take the gradient on V
        for mid in unique_movie:
            # simply take the first one if repeated
            midex = np.where(movies==mid)[0]
            uid = users[midex]
            V_UVterm = np.dot(self.U[:, uid], (UV_subset[midex] - ratings[midex]))
            Vterm = lambda_v*self.V[:, mid]
            V_SDG = V_UVterm + Vterm
            # update the U by the gradient
            self.V[:, mid] = self.V[:, mid] - alpha*V_SDG