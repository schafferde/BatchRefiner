#This is in it's own file to faciliate easier pickling for multiprocessing.
from scib.metrics.lisi import lisi_graph_py
from anndata import AnnData
from scanpy.pp import neighbors
from numpy import nanmedian


def ilisi_score(dim, batch_labels, n_neighbors=15, **kwargs):
    adata_tmp = AnnData(X=dim, obs={"batch":batch_labels})
    neighbors(adata_tmp, n_neighbors=n_neighbors, copy=False)
    ilisi_scores = lisi_graph_py(adata=adata_tmp, obs_key='batch', **kwargs) #n_cores=...
    ilisi = nanmedian(ilisi_scores)
    ilisi = (ilisi - 1)# / (adata.obs['batch'].nunique() - 1)
    return ilisi
