"""
Convolutional Neural Networks for Sentence Classification

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
"""

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
from theano.tensor.shared_randomstreams import RandomStreams
from collections import OrderedDict
import time
import cPickle as pickle

# ----------------------------------------------------------------------
# Activation functions

def ReLU(x):
    return T.maximum(0.0, x)


# def Sigmoid(x):
#     return T.nnet.sigmoid(x)


def Iden(x):
    return x
        

# ----------------------------------------------------------------------
class Layer(object):
  def __init__(self):
    self.params = []

  def output(self, input):
    raise NotImplementedError("Each concrete class needs to implement output")

  def __repr__(self):
    return "{}".format(self.__class__.__name__)

# ----------------------------------------------------------------------
class LinearLayer(Layer):
    """
    Class for LinearLayer
    """
    def __init__(self, rng, n_in, n_out, activation, W=None, b=None,
                 use_bias=False):

        self.activation = activation

        if W is None:            
            if activation.func_name == "ReLU":
                W_values = np.asarray(0.01 * rng.standard_normal(size=(n_in, n_out)),
                                      dtype=theano.config.floatX)
            else:                
                high = np.sqrt(6. / (n_in + n_out))
                W_values = np.asarray(rng.uniform(low=-high, high=high, size=(n_in, n_out)),
                                      dtype=theano.config.floatX)
            W = theano.shared(value=W_values, name='W')        
        if b is None and use_bias:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b')

        self.W = W
        self.b = b
    
        # parameters of the model
        if use_bias:
            self.params = [self.W, self.b]
        else:
            self.params = [self.W]


    def output(self, input):
        if self.b:
            output = T.dot(input, self.W) + self.b
        else:
            output = T.dot(input, self.W)

        if self.activation:
            output = self.activation(output)
        return output

# ----------------------------------------------------------------------
class DropoutLayer(Layer):

    def __init__(self, rng, dropout_rate):
        super(DropoutLayer, self).__init__()
        self.dropout_rate = dropout_rate
        self.srng = RandomStreams(rng.randint(1e6))

    def output(self, input):
        mask = self.srng.binomial(n=1, p=1-self.dropout_rate, size=input.shape,
                                  dtype=theano.config.floatX)
        return input * mask


# ----------------------------------------------------------------------
class LinearDropoutLayer(LinearLayer):

    def __init__(self, rng, n_in, n_out,
                 activation, dropout_rate, use_bias, W=None, b=None):
        super(LinearDropoutLayer, self).__init__(rng=rng, n_in=n_in, n_out=n_out, W=W, b=b,
                                           activation=activation, use_bias=use_bias)
        self.dropout_rate = dropout_rate
        self.srng = RandomStreams(rng.randint(1e6))

    def output(self, input):
        mask = self.srng.binomial(n=1, p=1-self.dropout_rate, size=input.shape,
                                  dtype=theano.config.floatX)
        return input * mask


# ----------------------------------------------------------------------
class MLPDropout(object):
    """A multilayer perceptron with dropout"""


    def __init__(self, rng, input, layer_sizes, dropout_rates, activations, use_bias=True):

        # Set up all the hidden layers
        self.weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        self.activations = activations
        next_layer_input = input
        # dropout the input
        next_dropout_layer = DropoutLayer(rng, dropout_rates[0])
        next_dropout_layer_input = next_dropout_layer.output(input)
        layer_index = 0
        for n_in, n_out in self.weight_matrix_sizes[:-1]:
            next_dropout_layer = LinearDropoutLayer(rng=rng,
                    activation=activations[layer_index],
                    n_in=n_in, n_out=n_out, use_bias=use_bias,
                    dropout_rate=dropout_rates[layer_index])
            self.dropout_layers.append(next_dropout_layer)
            next_dropout_layer_input = next_dropout_layer.output(next_dropout_layer_input)

            # Reuse the parameters from the dropout layer here, in a different
            # path through the graph.
            next_layer = LinearLayer(rng=rng,
                    activation=activations[layer_index],
                    # scale the weight matrix W by (1-p)
                    W=next_dropout_layer.W * (1 - dropout_rates[layer_index]),
                    b=next_dropout_layer.b,
                    n_in=n_in, n_out=n_out,
                    use_bias=use_bias)
            self.layers.append(next_layer)
            next_layer_input = next_layer.output(next_layer_input)
            layer_index += 1
        
        # Set up the output layer
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(input=next_dropout_layer_input,
                                                  n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)

        # Again, reuse paramters in the dropout output.
        output_layer = LogisticRegression(
            input=next_layer_input,
            # scale the weight matrix W by (1-p)
            W=dropout_output_layer.W * (1 - dropout_rates[-1]),
            b=dropout_output_layer.b,
            n_in=n_in, n_out=n_out)
        self.layers.append(output_layer)

        # output layer activation is softmax
        self.activations.append(T.nnet.softmax)

        # Use the negative log likelihood of the logistic regression layer as
        # the objective.
        self.dropout_negative_log_likelihood = self.dropout_layers[-1].negative_log_likelihood
        self.dropout_errors = self.dropout_layers[-1].errors

        self.negative_log_likelihood = self.layers[-1].negative_log_likelihood
        self.errors = self.layers[-1].errors

        # Note: the params in self.layers are not updated
        # Grab all the parameters together.
        self.params = [ param for layer in self.dropout_layers for param in layer.params ]


    def output(self, input):
        """
        :return: symbolic expression to predict results for :param input:
        """
        for i,layer in enumerate(self.layers):
            input = self.activations[i](T.dot(input, layer.W) + layer.b)
        return T.argmax(input, axis=1)


# ----------------------------------------------------------------------
class MLP(object):
    """
    Multi-Layer Perceptron.

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``LinearLayer`` class)  while the
    top layer is a softamx layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a LinearLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = LinearLayer(rng=rng,
                                       n_in=n_in, n_out=n_hidden,
                                       activation=T.tanh)

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output(input),
            n_in=n_hidden,
            n_out=n_out)

        # L1 norm; one regularization option is to enforce L1 norm to
        # be small

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = self.logRegressionLayer.negative_log_likelihood
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of its two layers
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        

# ----------------------------------------------------------------------
class LogisticRegression(object):
    """
    Multi-class Logistic Regression.

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W=None, b=None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)
    
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie
    
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        
        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            self.W = theano.shared(
                    value=np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    name='W')
        else:
            self.W = W

        # initialize the baises b as a vector of n_out 0s
        if b is None:
            self.b = theano.shared(
                    value=np.zeros((n_out,), dtype=theano.config.floatX),
                    name='b')
        else:
            self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

        \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
        \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
        \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
        correct label

        Note: we use the mean instead of the sum so that
        the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch ;
        zero one loss over the size of the minibatch
    
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
        correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        

# ----------------------------------------------------------------------
class LeNetConvPoolLayer(Layer):
    """Pool Layer of a convolutional network."""

    def __init__(self, rng, filter_shape, image_shape, poolsize=(2, 2), non_linear="tanh"):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: np.random.RandomState
        :param rng: a random number generator used to initialize weights

        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # (num output feature maps * filter height * filter width) / pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(poolsize)) # int div.
        # initialize weights with random weights
        if self.non_linear == "none" or self.non_linear == "relu":
            W_bound = 0.01
        else:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                                          dtype=theano.config.floatX),
                               borrow=True, name="W_conv")
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True, name="b_conv")
        
        self.params = [self.W, self.b]
        
    def output(self, input):
        """
        :return: symbolic expression to predict results for :param input:
        """
        #input_shape = (batch_size, 1, self.image_shape[2], self.image_shape[3])
        conv_out = conv2d(input=input, filters=self.W,
                          filter_shape=self.filter_shape,
                          input_shape=self.image_shape)

        pooled_out = downsample.max_pool_2d(input=conv_out, ds=self.poolsize, ignore_border=True)

        output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        if self.non_linear == "tanh":
            output = T.tanh(output)
        elif self.non_linear == "relu":
            output = ReLU(output)

        return output
        

# ----------------------------------------------------------------------
class EmbeddingLayer(Layer):
    """
    Embedding layer: word embeddings representations.
    """

    def __init__(self, vectors=None, name='embedding_layer'):
        """
        :param vectors: numpy array.
        """

        self.W = theano.shared(value=vectors, name=name)
        # Define parameters
        self.params = [self.W]

    def output(self, x):
        """
        Return the embeddings of the given indexes.
        :param x: tensor of shape (d1, d2)
        """
        return self.W[x]

# ---------------------------------------------------------------------
class ConvNet(MLPDropout):
    """
    Adds convolution layers in front of a MLPDropout.
    """


    def __init__(self, embeddings, height, filter_hs, conv_activation,
                 feature_maps, output_units, batch_size, dropout_rates,
                 activations=[Iden]):
        """
        :param embeddings: word embeddings
        :param height: sentence length (padded as necessary)
        :param filter_hs: filter window sizes    
        :param conv_activation: activation functin for the convolutional layer
        :param feature_maps: the size of feature maps (per filter window)
        :param output_units: number of output variables
        """
        rng = np.random.RandomState(3435)
        self.batch_size = batch_size

        # define model architecture
        self.index = T.lscalar() # minibatch number
        self.x = T.imatrix('x')  # a minibatch of words
        self.y = T.ivector('y')  # corresponding outputs

        width = embeddings.shape[1]

        self.emb_layer = EmbeddingLayer(embeddings, name="Words")

        # CHECKME: what for set_zero?
        zero_vec_tensor = T.vector()
        self.zero_vec = np.zeros(width)
        # set to 0 Words[0]
        Words = self.emb_layer.W
        self.set_zero = theano.function([zero_vec_tensor],
                                        updates=[(Words,
                                                  T.set_subtensor(Words[0,:],
                                                                  zero_vec_tensor))],
                                        allow_input_downcast=True)
        # inputs to the ConvNet go to all convolutional filters:
        image_shape = (batch_size, 1, height, width) # e.g. (50, 1, 66, 300)
        layer0_input = self.emb_layer.output(self.x).reshape(image_shape)
              #(self.x.shape[0], 1, self.x.shape[1], width))
        self.conv_layers = []
        # outputs of the convolutional filters
        layer1_inputs = []
        filter_w = width    
        for filter_h in filter_hs:
            filter_shape = (feature_maps, 1, filter_h, filter_w) # e.g. (100, 1, 7, 300)
            pool_size = (height-filter_h+1, 1) # e.g. (60, 1)
            conv_layer = LeNetConvPoolLayer(rng,
                                            image_shape=image_shape,
                                            filter_shape=filter_shape,
                                            poolsize=pool_size,
                                            non_linear=conv_activation)
            layer1_input = conv_layer.output(layer0_input).flatten(2)
            self.conv_layers.append(conv_layer)
            layer1_inputs.append(layer1_input)
        # inputs to the MLP
        layer1_input = T.concatenate(layer1_inputs, 1)
        layer_sizes = [feature_maps*len(filter_hs), output_units]
        # initiailze MLPDropout
        super(ConvNet, self).__init__(rng, input=layer1_input,
                                      layer_sizes=layer_sizes,
                                      activations=activations,
                                      dropout_rates=dropout_rates)

        # add embeddings
        self.params += self.emb_layer.params
        # add parameters from convolutional layers
        for conv_layer in self.conv_layers:
            self.params += conv_layer.params


    def output(self, input):
        """
        :return: symbolic expression to predict results for data in :param input:
        """
        layer0_input = self.emb_layer.output(self.x).reshape(
            (self.x.shape[0], 1, self.x.shape[1], self.emb_layer.W.shape[1]))
        layer0_outputs = [ layer.output(layer0_input, input.shape[0]).flatten(2)
                           for layer in self.conv_layers ]
        mlp_input = T.concatenate(layer0_outputs, 1)
        return super(ConvNet, self).output(mlp_input) # MLPDropout


    def train(self, train_set, shuffle_batch=True,
              epochs=25, updater=None, save=lambda: None):
        """
        Train a simple conv net
        :param train_set: list of word indices, last one is y.
        :param shuffle_batch: whether to shuffle mini-batches.
        :param updater: update algorithm.
        :param save: function for saving the model.
        """

        # same as using the combination softmax_cross_entropy_with_logits in tf
        cost = self.negative_log_likelihood(self.y) 
        dropout_cost = self.dropout_negative_log_likelihood(self.y)           
        # updates: list of (variable, delta)
        updates = updater.updates(dropout_cost, self.params)

        # shuffle dataset and assign to mini batches.
        # if dataset size is not a multiple of batch size, replicate 
        # extra data (at random)
        np.random.seed(3435)    # DEBUG
        batch_size = self.batch_size
        if train_set.shape[0] % batch_size > 0:
            extra_data_num = batch_size - train_set.shape[0] % batch_size
            #extra_data = train_set[np.random.choice(train_set.shape[0], extra_data_num)]
            perm_set = np.random.permutation(train_set)   
            extra_data = perm_set[:extra_data_num]
            new_data = np.append(train_set, extra_data, axis=0)
        else:
            new_data = train_set
        shuffled_data = np.random.permutation(new_data) # Attardi
        n_batches = shuffled_data.shape[0] // batch_size

        # divide train batches into 90% train, 10% validation batches
        n_train_batches = int(np.round(n_batches*0.9))
        n_val_batches = n_batches - n_train_batches
        train_set = shuffled_data[:n_train_batches*batch_size,:]
        val_set = shuffled_data[n_train_batches*batch_size:,:]     

        # y are stored in train_set[:-1]
        train_set_x = theano.shared(train_set[:,:-1], borrow=True)
        train_set_y = theano.shared(train_set[:,-1], borrow=True)
        batch_start = self.index * batch_size
        batch_end = batch_start + batch_size
        # compile Theano functions to get train/val errors
        train_function = theano.function([self.index], cost, updates=updates,
                                         givens={
                                             self.x: train_set_x[batch_start:batch_end],
                                             self.y: train_set_y[batch_start:batch_end]},
                                         allow_input_downcast = True)

        # errors on train set
        train_error = theano.function([self.index], self.errors(self.y),
                                      givens={
                                          self.x: train_set_x[batch_start:batch_end],
                                          self.y: train_set_y[batch_start:batch_end]},
                                      allow_input_downcast=True)

        # errors on val set
        val_set_x = theano.shared(val_set[:,:-1], borrow=True)
        val_set_y = theano.shared(val_set[:,-1], borrow=True)
        val_error = theano.function([self.index], self.errors(self.y),
                                    givens={
                                        self.x: val_set_x[batch_start:batch_end],
                                        self.y: val_set_y[batch_start:batch_end]},
                                    allow_input_downcast=True)

        # start training over mini-batches
        print 'training...'
        best_val_perf = 0
        for epoch in xrange(epochs):
            start_time = time.time()
            # FIXME: should permute whole set rather than minibatch indexes
            if shuffle_batch:
                for minibatch_index in np.random.permutation(range(n_train_batches)):
                    cost_epoch = train_function(minibatch_index)
                    self.set_zero(self.zero_vec) # CHECKME: Why?
            else:
                for minibatch_index in xrange(n_train_batches):
                    cost_epoch = train_function(minibatch_index)  
                    self.set_zero(self.zero_vec)
            train_losses = [train_error(i) for i in xrange(n_train_batches)]
            train_perf = 1 - np.mean(train_losses)
            val_losses = [val_error(i) for i in xrange(n_val_batches)]
            val_perf = 1 - np.mean(val_losses)                        

            print('epoch: %i, training time: %.2f secs, train perf: %.2f %%, val perf: %.2f %%' % (
                epoch, time.time()-start_time, train_perf * 100,
                val_perf * 100))
            if val_perf >= best_val_perf:
                best_val_perf = val_perf
                # dump model
                save()
        return val_perf


    def __getvalues__(self):
        """Access value of parameters."""
        return [p.get_value() for p in self.params]


    def __setvalues__(self, weights):
        """Set the value of parameters."""
        for p,w in zip(self.params, weights):
            p.set_value(w)


    def save(self, mfile):
        """
        Save network params to file.
        """
        pickle.dump((self.params, self.layers, self.emb_layer, self.conv_layers, self.activations),
                    mfile, protocol=pickle.HIGHEST_PROTOCOL)
        # FIXME: we might dump values and recreate layers.
        #pickle.dump(self.__getvalues__(), mfile, protocol=pickle.HIGHEST_PROTOCOL)


    @classmethod
    def load(cls, mfile):
        cnn = cls.__new__(cls)
        cnn.params, cnn.layers, cnn.emb_layer, cnn.conv_layers, cnn.activations = pickle.load(mfile)
        cnn.x = T.imatrix('x')   
        return cnn


