import numpy as np
class Optimizer:
    def __init__(self, alpha, beta1, beta2, epsilon = 1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
    def update(self, theta, grad):
        if np.isscalar(theta):
            theta = np.array([theta], dtype=np.float64)
            grad = np.array([grad], dtype=np.float64)
        if self.m is None:
            self.m = np.zeros(theta.shape)
            self.v = np.zeros(theta.shape)
        self.t+=1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)
        alphat = self.alpha * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        theta-=  (alphat * self.m) / (np.sqrt(self.v) + self.epsilon)
        if theta.size==1:
            return theta.item()
        return theta
class Model:
    def __init__(self, alpha, beta1, beta2, epochs):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.weights_optimizer = Optimizer(alpha = self.alpha,
        beta1 = self.beta1, beta2= self.beta2)
        self.bias_optimizer = Optimizer(alpha = self.alpha,
        beta1=self.beta1, beta2 = self.beta2)


    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    def fit(self, x, y):
       m = x.shape[0]
       n_features = x.shape[1]
       y = y.reshape(-1,1)
       self.weights = np.random.randn(n_features, 1)
       self.bias = 0
       self.history = []
       self.losses = []
       #prediction.
       for i in range(0, self.epochs + 1):
           z = np.dot(x, self.weights) + self.bias
           epsilon = 1e-15
           y_hat = np.clip(self.sigmoid(z), epsilon, 1-epsilon)
           dw = (1/m) * np.dot(x.T, (y_hat - y))
           db = (1/m) * np.sum(y_hat - y)
           self.weights = self.weights_optimizer.update(self.weights, dw)
           self.bias = self.bias_optimizer.update(self.bias, db)
           current_score = self.score(x, y)
           self.history.append(current_score)
           loss = -np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat))
           self.losses.append(loss)
    def predict_proba(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.sigmoid(z)


    def score(self, x, y):
        proba= self.predict_proba(x)
        y_pred = (proba >=0.5).astype(int)
        y_true = np.array(y).reshape(-1, 1)
        correct = np.sum(y_pred == y_true)
        total = len(y_true)
        return correct / total




