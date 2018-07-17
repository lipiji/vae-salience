#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from updates import *
from attention_soft import *
from attention_dot import *

class VAE(object):
    def __init__(self, in_size, out_size, hidden_size, latent_size, \
            sent_size, num_summs, optimizer = "adadelta"):
        self.prefix = "VAE_"
        self.X = T.matrix("X")
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.optimizer = optimizer
        self.sent_size = sent_size
        self.num_sents = sent_size
        self.num_summs = num_summs
        #self.para = T.matrix("para")

        self.define_layers()
        self.define_train_test_funcs()
        
    def define_layers(self):
        self.params = []
        
        layer_id = "1"
        self.W_xh = init_weights((self.in_size, self.hidden_size), self.prefix + "W_xh" + layer_id)
        self.b_xh = init_bias(self.hidden_size, self.prefix + "b_xh" + layer_id)

        layer_id = "2"
        self.W_hu = init_weights((self.hidden_size, self.latent_size), self.prefix + "W_hu" + layer_id)
        self.b_hu = init_bias(self.latent_size, self.prefix + "b_hu" + layer_id)
        self.W_hsigma = init_weights((self.hidden_size, self.latent_size), self.prefix + "W_hsigma" + layer_id)
        self.b_hsigma = init_bias(self.latent_size, self.prefix + "b_hsigma" + layer_id)

        layer_id = "3"
        self.W_zh = init_weights((self.latent_size, self.hidden_size), self.prefix + "W_zh" + layer_id)
        self.b_zh = init_bias(self.hidden_size, self.prefix + "b_zh" + layer_id)
   
        layer_id = "4"
        self.W_hy = init_weights((self.hidden_size, self.out_size), self.prefix + "W_hy" + layer_id)
        self.b_hy = init_bias(self.out_size, self.prefix + "b_hy" + layer_id)

        layer_id = "s"
        self.Pos = init_weights((self.num_summs, self.latent_size), self.prefix + "Pos" + layer_id, sample="uniform")
        #self.W_pz = init_weights((self.num_sents, self.num_summs), self.prefix + "W_pz" + layer_id)
        #self.W_sx = init_weights((self.num_sents, self.num_summs), self.prefix + "W_sx" + layer_id)

        self.params += [self.W_xh, self.b_xh, self.W_hu, self.b_hu, self.W_hsigma, self.b_hsigma, \
                        self.W_zh, self.b_zh, self.W_hy, self.b_hy, self.Pos]#, self.W_pz, self.W_sx]

        # encoder
        h_enc = T.nnet.relu(T.dot(self.X, self.W_xh) + self.b_xh)
        self.H = h_enc

        self.mu = T.dot(h_enc, self.W_hu) + self.b_hu
        log_var = T.dot(h_enc, self.W_hsigma) + self.b_hsigma
        self.var = T.exp(log_var)
        self.sigma = T.sqrt(self.var)

        srng = T.shared_randomstreams.RandomStreams(234)
        eps = srng.normal(self.mu.shape)
        self.z = self.mu + self.sigma * eps

        # decoder
        h_dec = T.nnet.relu(T.dot(self.z, self.W_zh) + self.b_zh)
        self.reconstruct = T.nnet.sigmoid(T.dot(h_dec, self.W_hy) + self.b_hy)

        # point mining
        #attentin_soft_z = SoftAttentionLayer(layer_id + "a1", (self.num_summs, self.num_sents, self.latent_size), self.z, self.Pos) 
        #self.params += attentin_soft_z.params
        #self.Az = attentin_soft_z.A
        #z_dec_summ_a = attentin_soft_z.activation
        
        h_dec_summ = T.nnet.relu(T.dot(self.Pos, self.W_zh) + self.b_zh)
        attentin_soft_h = SoftAttentionLayer(layer_id + "a2", (self.num_summs, self.num_sents, self.hidden_size), h_enc, h_dec_summ) 
        self.params += attentin_soft_h.params
        self.Ah = attentin_soft_h.A
        h_dec_summ_a = attentin_soft_h.activation
        
        s_summ = T.nnet.sigmoid(T.dot(h_dec_summ_a, self.W_hy) + self.b_hy)
        attentin_dot = DotAttentionLayer(layer_id, (self.num_summs, self.num_sents, self.out_size), self.X, s_summ)
        self.params += attentin_dot.params 
        self.Ax = attentin_dot.A

        self.hidden_summs = attentin_dot.activation

        self.reconstruct_z = T.dot(self.Ax.T, self.Pos)
        self.reconstruct_h = T.dot(self.Ax.T, h_dec_summ_a)
        self.reconstruct_x = T.dot(self.Ax.T, self.hidden_summs)

    def multivariate_bernoulli(self, y_pred, y_true):
        return T.sum(y_true * T.log(y_pred) + (1 - y_true) * T.log(1 - y_pred), axis=1)

    def kld(self, mu, var):
        return 0.5 * T.sum(1 + T.log(var) - mu**2 - var, axis=1)
    
    def cost_summary_hidden(self, pred, label):
        cost = []
        for j in xrange(0, self.num_summs):
            yj = pred[j, :]
            y = T.repeat(T.reshape(yj, (1, pred.shape[1])), label.shape[0], axis=0)
            cost.append(self.w_cost_mse(y, label))
        return T.mean(cost)

    def cost_mse(self, pred, label):
        #cost = T.mean((pred - label) ** 2)
        mse = T.mean((pred - label) ** 2, axis=1)
        cost = T.sum(mse)
        return cost

    def w_cost_mse(self, pred, label):
        mse = T.mean((pred - label) ** 2, axis=1)
        mse = T.reshape(mse, (1, self.num_sents))
        pos = T.reshape(self.para, (self.num_sents, 1))
        cost = T.sum(T.dot(mse, pos))
        return cost

    def define_train_test_funcs(self):
        a = self.kld(self.mu, self.var)
        b = self.multivariate_bernoulli(self.reconstruct, self.X)
        c =   self.cost_mse(self.reconstruct_z, self.z)
        d =  400 * self.cost_mse(self.reconstruct_h, self.H)
        e =  800 * self.cost_mse(self.reconstruct_x, self.X)
        cost = -T.mean(a + b) + c + d + e

        gparams = []
        for param in self.params:
            #gparam = T.grad(cost, param)
            gparam = T.clip(T.grad(cost, param), -10, 10)
            gparams.append(gparam)

        lr = T.scalar("lr")
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)
        
        self.train = theano.function(inputs = [self.X, lr], \
                outputs = [cost, a, b, c, d, e,  self.hidden_summs,  self.Ax], updates = updates)
        #self.validate = theano.function(inputs = [self.X], outputs = [cost, self.reconstruct])
        #self.project = theano.function(inputs = [self.X], outputs = self.mu)
        #self.generate = theano.function(inputs = [self.z], outputs = self.reconstruct)
  
