from .mlp import MLP

class Estimator(MLP):
    def __init__(self, 
                 input_dim, 
                 output_dim, 
                 hidden_dim=[256, 128], 
                 activation="relu",
                 lr = 1e-4,
                 **kwargs):
        super(Estimator, self).__init__(input_dim, output_dim, hidden_dim, activation, lr, None, None, **kwargs)