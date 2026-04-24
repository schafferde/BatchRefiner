#This is in it's own file to faciliate easier pickling for multiprocessing.
from scib.metrics import pc_regression 

def r2_score(dim, batch_labels, **kwargs):  
    return pc_regression(dim, batch_labels, **kwargs)

