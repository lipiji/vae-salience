#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *

class SoftAttentionLayer(object):
    def __init__(self, layer_id, shape, sent_encs, sent_decs):
        prefix = "AttentionLayer_"
        layer_id = "_" + layer_id
        self.num_summs, self.num_sents, self.out_size = shape
        
        self.W_a1 = init_weights((self.out_size, self.out_size), prefix + "W_a1" + layer_id)
        self.W_a2 = init_weights((self.out_size, self.out_size), prefix + "W_a2" + layer_id)
        self.W_a3 = init_weights((1, self.out_size), prefix + "W_a3" + layer_id)
        self.W_a4 = init_weights((self.out_size, self.out_size), prefix + "W_a4" + layer_id)
        #self.W_a5 = init_weights((self.num_summs, 1), prefix + "W_a5" + layer_id)
        #self.W_a6 = init_weights((self.num_summs, 1), prefix + "W_a6" + layer_id)
        self.W_a5 = init_weights((self.out_size, self.out_size), prefix + "W_a5" + layer_id)

        def attend(h, sent_encs):
            h = T.reshape(h, (1, self.out_size))
            h_ = T.repeat(h, self.num_sents, axis=0)
            M = T.tanh(T.dot(self.W_a1, sent_encs.T) + T.dot(self.W_a2, h_.T))
            a = T.nnet.softmax(T.dot(self.W_a3, M))
            return a
        outputs, updates = theano.scan(attend, sequences = sent_decs, non_sequences = sent_encs)

        A = T.reshape(outputs, (self.num_summs, self.num_sents))

        c = T.dot(A, sent_encs)

        # new sentence codes
        #self.activation = T.tanh(T.dot(sent_decs, self.W_a4)) * T.repeat(self.W_a5, self.out_size, axis=1) \
        #                  + c * T.repeat(self.W_a6, self.out_size, axis=1)
        
        self.activation = T.tanh(T.dot(c, self.W_a4) + T.dot(sent_decs, self.W_a5))

        self.A = A
        self.params = [self.W_a1, self.W_a2, self.W_a3, self.W_a4, self.W_a5]
        #self.params = [self.W_a4, self.W_a5, self.W_a6]
