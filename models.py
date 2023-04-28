

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import Tensor, Parameter, Model 
from mindspore.common.initializer import initializer, XavierUniform
import mindspore as ms
from mindspore import ops




class GATLayer(nn.Cell):
    """
    Simple MindSpore Implementation of the Graph Attention layer.
    """
    def __init__(self,in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        
        self.dropout       = dropout        # drop prob = 0.6
        self.in_features   = in_features    # 
        self.out_features  = out_features   # 
        self.alpha         = alpha          # LeakyReLU with negative input slope, alpha = 0.2
        self.concat        = concat         # conacat = True for all layers except the output layer.
        gain = 1.414
        # Initialize the weight matrix W
        self.attn_s = ms.Parameter(initializer(XavierUniform(gain), [in_features, out_features], ms.float32), name="attn_s_{}".format(out_features))
        self.attn_d = ms.Parameter(initializer(XavierUniform(gain), [2 * out_features,1], ms.float32), name="attn_d_{}".format(out_features))
        
        #self.attn_s = Parameter(initializer(XavierUniform(gain), [num_attn_head, out_size], ms.float32), name="attn_s")
        #self.attn_d = Parameter(initializer(XavierUniform(gain), [num_attn_head, 1], ms.float32),name="attn_d")
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def construct(self, input, adj):
        # Linear Transformation
        h = P.MatMul()(input, self.attn_s)
        #N=h.shape[0]
        
        # Attention Mechanism
        # Perform the concatenation operation
        #a_input = P.Concat(1)((P.Reshape()(P.Tile()(h, (1, N)), (N * N, -1)), P.Tile()(h, (N, 1))))
        #a_input = P.Reshape()(a_input, (N, -1, 2 * self.out_features))
        #e = self.leakyrelu(ops.matmul(a_input, self.attn_d).squeeze(2))

        e = self._prepare_attentional_mechanism_input(h)
        # Masked Attention 
        zero_vec = -9e15*ops.ones_like(e)
        attention = ops.where(adj > 0, e, zero_vec)
        
        softmax = ops.Softmax(axis=1)
        attention = softmax(attention)
        #dropout = nn.Dropout(p = self.dropout)
        attention = ops.dropout(attention, p = self.dropout)

        h_prime   = ops.matmul(attention, h)
        #elu= nn.ELU()
        if self.concat:
            return ops.elu(h_prime)
        else:
            return h_prime
    def _prepare_attentional_mechanism_input(self,h):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        h1 = ops.matmul(h, self.attn_d[:self.out_features, :])
        h2 = ops.matmul(h, self.attn_d[self.out_features:, :])
        # broadcast add
        e = h1 + h2.T
        return self.leakyrelu(e)
        




class MultiHeadGATLayer(nn.Cell):
    """
    MindSpore Implementation of the Multi-head Graph Attention layer.
    """
    def __init__(self, input_feature_size, output_size, nclass, dropout, alpha, nheads):
        super(MultiHeadGATLayer, self).__init__()
        self.dropout=dropout
        
        
        #self.attentions=[GATLayer(input_feature_size, output_size, dropout=dropout, alpha=alpha, concat=False)
        #               for _ in range(nheads)]
        
        #for i, attention in enumerate(self.attentions):
        #   self.insert_child_to_cell('attention_{}'.format(i),attention)
            
        self.attentions = nn.CellList()
        for _ in range(nheads):
            attention = GATLayer(in_features= input_feature_size, out_features=output_size, dropout=dropout, alpha=alpha, concat=False)
            self.attentions.append(attention)
        
        self.out_att = GATLayer(in_features= output_size*nheads, out_features = nclass, dropout=dropout, alpha=alpha, concat=True)

    def construct(self, x, adj):
        
        dropout = nn.Dropout(p = self.dropout)
        #dropout = ops.dropout(p = self.dropout)
        x=dropout(x)
        x=ops.cat([att(x,adj) for att in self.attentions], axis = 1)
        x=dropout(x)
        elu = nn.ELU()
        x= elu(self.out_att(x,adj))
        
        return ops.log_softmax(x, axis = 1)







