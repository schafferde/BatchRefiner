import os
import warnings
from multiprocessing import Pool
import numpy as np
from functools import partial


def batchrefine(embed, batch_labels, mode="scale", metric="pcr", keep_scores=False, filter_dims=50, n_proc=8, **kwargs):
    """
    Apply BatchRefiner to an embedding

    Parameters
    ----------
    data : numpy.ndarray | scipy.sparse.* | ...
        Embeddings of shape (cells, dimensions).
    batch_labels : numpy.ndarray | list | ...
        Batch labels of shape (cells, ).
    mode : str, optional
        BatchRefiner mode to use. Defaults to "scale"; "filter" is also supported.
    metric : str | callable, optional
        Metric for scoring dimensions. PCR Comparison ("pcr", default) and iLISI ("ilisi") are implemented using scib. 
        Otherwise, a user-supplied funciton is used. The function must take an embedding, an array of batch labels, 
        and optionally additional kwargs. 
    keep_scores : bool, optional
        If True, returns scores (dimensions,) along with the results.
    filter_dims : bool, optional
        In "filter" mode, number of dimemsions to keep. Defaults to 50.
    n_proc : int, optional
        Number of parallel processes to use for scoring dimensions. 
        Defaults to 8; -1 will specify the number of available CPUs.
    **kwargs : dict
        Additional arguments to be passed to the dimension scoring function, such as n_neighbors for iLISI.

    Returns
    -------
    If keep_scores is False, BatchRefiner-modified embeddings with shape (cells, dimensions) in "scale" mode
        or (cells, filter_dims) in "filter" mode.
    If keep_scores is True, a tuple of modified embeddings and an ndarray of scores with shape (dimensions,).
    """
    #Validate input
    _, n_dims = embed.shape
    if mode == "filter":
        if filter_dims >= n_dims:
            warnings.warn(
                f"Requested filtered dimensions {filter_dims} match or exceed the number of inital dimensions {n_dims}.",
                RuntimeWarning,
            )
            return embed
    elif mode != "scale":
        raise ValueError(f"Invalid mode {mode} specified. Supported modes are 'scale' or 'filter'.")
    
    #set up scoring
    if callable(metric):
        #Using user-supplied benchmarking function.
        score_fn = metric
    elif metric == "pcr":
        from .pcr_score import pcr_score
        score_fn = pcr_score
    elif metric == "ilisi":
        from .ilisi_score import ilisi_score
        score_fn = ilisi_score
    else:
        raise ValueError(f"Invalid metric {metric} specified. Supported modes are 'pcr', 'ilisi', or a user-provided function.")
        
    partial_score = partial(score_fn, batch_labels=batch_labels, **kwargs)
    dimensions = [embed[:,i].reshape(-1,1) for i in range(n_dims)]
    if n_proc == -1:
        n_proc = os.cpu_count()
    n_proc = min(os.cpu_count(), n_proc)

    #Compute scores
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        if n_proc == 1:
            scores = np.asarray([partial_score(d) for d in dimensions])
        else:
            with Pool(n_proc) as p:
                scores = np.asarray(p.map(partial_score, dimensions))

    #BatchRefiner
    if mode == "scale":
        scores -= np.min(scores)
        max_val = np.max(scores)
        scores /= max_val if max_val > 0 else 1 #Becomes a no-op if all the same
        result = embed * scores
    else:
        columns = np.argpartition(scores, -filter_dims)[-filter_dims:]
        result = embed[:,columns]
    
    if keep_scores:
        return (result, scores)
    else:
        return result

def batchrefine_scanpy(adata, emb_key, batch_key="batch", br_key="X_BatchRefiner", score_key=None, copy=False, **kwargs):
    """
    Apply BatchRefiner to an embedding in an AnnData object

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData containing embedding and metadata
    emb_key : str
        Key of the input embeddings, with shape (cells, dimensions), in adata.obsm. 
    batch_labels : str, optional
        Key of the batch labels in adata.obs. Default is "batch".
    mode : str, optional
        BatchRefiner mode to use. Defaults to "scale"; "filter" is also supported.
    metric : str | callable, optional
        Metric for scoring dimensions. PCR Comparison ("pcr", default) and iLISI ("ilisi") are implemented using scib. 
        Otherwise, a user-supplied funciton is used. The function must take an embedding, an array of batch labels, 
        and optionally additional kwargs. 
    br_key : 
        Key to save BatchRefiner-modified embeddings in adata.obsm. 
        These will have shape (cells, dimensions) in "scale" mode or (cells, filter_dims) in "filter" mode.
    score_key : str, optional
        Key to save the scores, with shape (dimensions,), in adata.uns. Default is to not save scores.
    filter_dims : int, optional
        In "filter" mode, number of dimemsions to keep. Defaults to 50.
    n_proc : int, optional
        Number of parallel processes to use for scoring dimensions.
        Defaults to 8; -1 will specify the number of available CPUs.
    copy : bool, optional
        If True, copy the AnnData object instead of (default) modifying in place.
    **kwargs : dict
        Additional arguments to be passed to the dimension scoring function, such as n_neighbors for iLISI.

    Returns
    -------
    If copy is False, None; the AnnData is modified in-place.
    If copy is True, a copy of the AnnData with BatchRefiner-modified embeddings, and scores if score_key is provided, added.
    """
    save_scores = score_key is not None
    result = batchrefine(adata.obsm[emb_key], adata.obs[batch_key], keep_scores=save_scores, **kwargs)
    if copy:
        adata = adata.copy()
    if save_scores:
        result, scores = result
        adata.uns[score_key] = scores
    adata.obsm[br_key] = result
    if copy:
        return adata
