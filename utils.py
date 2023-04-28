
import numpy as np
import scipy.sparse as sp
import mindspore
from mindspore import Tensor

from mindspore.ops import operations as P
from mindspore import dtype as mstype

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    
    classes = sorted(list(set(labels)))
    
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    
    return labels_onehot

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))    
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()  
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)   #D^{-0.5}
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)
    
def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))    
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    # Find the indices of the maximum values along dimension 1 (columns)
    preds = P.Argmax(axis=1)(output)

    # Cast the predictions tensor to the same data type as the labels tensor
    cast = P.Cast()
    preds = cast(preds, labels.dtype)

    # Compute element-wise equality between predictions and labels
    correct = P.Equal()(preds, labels)

    # Cast the correct tensor to float and sum the correct predictions
    correct = cast(correct, mstype.float32)
    correct_sum = P.ReduceSum()(correct)

    # Compute the accuracy by dividing the sum of correct predictions by the number of labels
    accuracy = correct_sum / len(labels)

    return accuracy

def load_data(path, dataset):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)} 
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32) 
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features) 
    adj = normalize_adj(adj + sp.eye(adj.shape[0])) #adj = D^{-0.5}SD^{-0.5}, S=A+I

    idx_train = np.arange(140)
    idx_val = np.arange(200, 500)
    idx_test = np.arange(500, 1500)

    adj = Tensor(np.array(adj.todense()), dtype=mindspore.float32)
    features = Tensor(np.array(features.todense()), dtype=mindspore.float32)
    labels = Tensor(np.where(labels)[1], dtype=mindspore.int32)

    idx_train = Tensor(idx_train, dtype=mindspore.int64)
    idx_val = Tensor(idx_val, dtype=mindspore.int64)
    idx_test = Tensor(idx_test, dtype=mindspore.int64)

    return adj, features, labels, idx_train, idx_val, idx_test







