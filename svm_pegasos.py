import numpy as np

# Abstract model object
class Model(object):

    def __init__(self, nfeatures, lmbda):
        self.num_input_features = nfeatures
        self.lmbda = lmbda

    # Fitting the model
    def fit(self, *, X, y):
        """ Fit.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """
        raise NotImplementedError()

    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """
        #raise NotImplementedError()

    # Fixes some feature disparities between datasets
    # Call this before performing inference to
    # make sure X features match the weights
    def _fix_test_feats(self, X):

        num_examples, num_input_features = X.shape
        if num_input_features < self.num_input_features:
            X = X.copy()
            X._shape = (num_examples, self.num_input_features)
        if num_input_features > self.num_input_features:
            X = X[:, :self.num_input_features]
        return X

class Pegasos(Model):

    def __init__(self, *, nfeatures, lmbda):
        """
        Args:
            nfeatures: size of feature space
            lmbda: regularizer term (lambda)

        Sets:
            W: weight vector
            t: current iteration
        """
        super().__init__(nfeatures=nfeatures, lmbda=lmbda)
        self.W = np.zeros((nfeatures, 1))
        self.t = 1

    def fit(self, *, X, y):
        """ Fit the model.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].
            y: A dense array of ints with shape [num_examples].
        """

        # Compute y_hat's by using the predict function and
        # convert x from sparse row matrix to numpy array
        y_hat = self.predict(X)
        X = X.toarray()

        for i in range(0, len(X)):

            # Convert class labels
            if y[i] == 0:
                y[i] = -1
            elif y[i] == 1:
                y[i] = 1

            # Obtain values for indicator function,
            # weight coefficient, and learning rate
            dot_prod = np.dot(self.W.T, X[i].T)
            if y[i] * dot_prod < 1:
                indicator = 1
            else:
                indicator = 0

            coeff_w_t = (1.0 - 1.0 / self.t)
            lr_t = 1.0 / (self.lmbda * self.t)

            # Weight update step
            update = coeff_w_t * self.W.T + lr_t * indicator * y[i] * X[i]
            update = update.T
            self.W += update

            # Increment time step
            self.t += 1

            # Gradient projection after each time step t to prevent overfitting:
            # Further regularize learned parameters by limiting the set of
            # candidate solutions to a ball of radius 1 / sqrt(λ)
            self.W = min(1, (1/ (self.lmbda**1/2))/np.linalg.norm(self.W) ) * self.W


    def predict(self, X):
        """ Predict.

        Args:
            X: A compressed sparse row matrix of floats with shape
                [num_examples, num_features].

        Returns:
            A dense array of ints with shape [num_examples].
        """

        # Ensure x features match the weights, then
        # convert to array
        X = self._fix_test_feats(X)
        X = X.toarray()
        y_hat = []

        # Make predictions of y_hat
        # if X[i] * W >= 0, then y_hat = 1
        for i in range(0, len(X)):
            m = np.dot(X[i], self.W)
            margin = np.asscalar(m)

            if margin >= 0:
                y_hat.append(1)
            else:
                y_hat.append(0)
        return y_hat
