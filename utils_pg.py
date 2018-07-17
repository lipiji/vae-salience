#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_normal_weight(shape, scale=0.01):
    return np.random.normal(loc=0.0, scale=scale, size=shape)

def init_uniform_weight(shape, scale=0.1):
    return np.random.uniform(-scale, scale, shape)

def init_xavier_weight_uniform(shape):
    return np.random.uniform(-np.sqrt(6. / (shape[0] + shape[1])), np.sqrt(6. / (shape[0] + shape[1])), shape)

def init_xavier_weight(shape):
    fan_in, fan_out = shape
    s = np.sqrt(2. / (fan_in + fan_out))
    return init_normal_weight(shape, s)

def init_ortho_weight(shape):
    W = np.random.normal(0.0, 1.0, (shape[0], shape[0]))
    u, s, v = np.linalg.svd(W)
    return u


def init_weights(shape, name, sample = "xavier"):
    if sample == "uniform":
        values = np.random.uniform(-0.08, 0.08, shape)
    elif sample == "xavier":
        values = np.random.uniform(-np.sqrt(6. / (shape[0] + shape[1])), np.sqrt(6. / (shape[0] + shape[1])), shape)
    elif sample == "ortho":
        W = np.random.randn(shape[0], shape[0])
        u, s, v = np.linalg.svd(W)
        values = u
    else:
        raise ValueError("Unsupported initialization scheme: %s" % sample)
    
    return theano.shared(floatX(values), name)

def init_gradws(shape, name):
    return theano.shared(floatX(np.zeros(shape)), name)

def init_bias(size, name):
    return theano.shared(floatX(np.zeros((size,))), name)

def init_mat(mat, name):
    return theano.shared(floatX(mat), name)

def save_model(f, model):
    ps = {}
    for p in model.params:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, "wb"))

def load_model(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params:
        p.set_value(ps[p.name])
    return model
