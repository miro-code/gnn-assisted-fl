
#import numpy as np
#import scipy.sparse as sp

#def normalize_adj(mx):
#    """Row-normalize sparse matrix"""
#    rowsum = np.array(mx.sum(1))
#    r_inv = np.power(rowsum, -1).flatten()
#    r_inv[np.isinf(r_inv)] = 0.
#    r_mat_inv = sp.diags(r_inv)
#    mx = r_mat_inv.dot(mx)
#    return mx
