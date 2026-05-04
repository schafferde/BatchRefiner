# BatchRefiner
BatchRefiner is a general postprocessing tool to boost batch integration of scRNA-seq cell embeddings. It uses metrics of batch inetegration to score individual dimensions of an embedding, and then down-weights (scales) or filters out embeddings with a high batch signal (poor score). It is implemented in Python.

## Installation

BatchRefiner can be installed from GitHub
```
git clone https://github.com/schafferde/BatchRefiner.git
cd BatchRefiner
pip install .
```
or simply:
```
pip install git+https://github.com/schafferde/BatchRefiner.git
```

### Dependencies
BatchRefiner requires the following packages:

* numpy
* anndata
* scanpy
* pandas
* [scib](https://github.com/theislab/scib)


## Example
An example workflow using BatchRefiner is provided in `example/example_pancreas.ipynb`. Here, we use Batch $R^2$ to scale PCA embeddings of a small dataset. The dataset includes log-normalized expression data of 8469 pancreatic alpha and beta cells from five batches. These data were normalized and then subset from a [larger dataset](https://zenodo.org/records/7968485) (Hie et al., 2024). The subset dataset is also found in `example/`.

## Usage
BatchRefiner postprocesses an existing cell embedding. This package supports directly imputting matrices or providing an AnnData object, used by Scanpy.

### Matrix input
The `batchrefine` function accepts a (cells) x (dimensions) matrix `embed`, and outputs a postprocessed version. The input matrix may be a Numpy `ndarray`, Scipy `csr_matrix`, or other array daya type. This function implements scaling, centering, and filtering modes, and both batch $R^2$ and iLISI scoring metrics. Additional metrics can be provided as a user-specified scoring function. 
```
from batchrefiner import batchrefine
#Input data
embed = ... #shape: (cells, dimensions)
batch_labels = ... #shape: (cells, )

#Run BatchRefiner
#Running scaling, using batch R^2 with 8 processes. The R^2 calculation has some internal parallelization, so a low n_proc is suggested.
scaled_pca = batchrefine(embed, batch_labels, mode="scale", metric="r2", n_proc=8)
#Running filtering, using iLISI with (# of CPUs) processes
filtered_ilisi = batchrefine(embed, batch_labels, mode="filter", metric="ilisi", n_proc=-1)
```
The docstring provides all options, including parallelism (``n_proc``), custom scoring functions (``metric=callable``), and saving scores (``keep_scores``)
```
>>> help(batchrefine)
    Apply BatchRefiner to an embedding

    Parameters
    ----------
    data : numpy.ndarray | scipy.sparse.* | ...
        Embeddings of shape (cells, dimensions).
    batch_labels : numpy.ndarray | list | ...
        Batch labels of shape (cells, ).
    mode : str, optional
        BatchRefiner mode to use. Defaults to "scale"; "filter" or "scale" are also supported.
    metric : str | callable, optional
        Metric for scoring dimensions. Batch R^2 ("r2", default) and iLISI ("ilisi") are implemented using scib. 
        Otherwise, a user-supplied funciton is used. The function must take an embedding, an array of batch labels, 
        and optionally additional kwargs. It should return a higher score for columns with more batch signal.
    keep_scores : bool, optional
        If True, returns scores (dimensions,) along with the results.
    filter_dims : int, optional
        In "filter" mode, number of dimemsions to keep. 
    filter_thresh : float, optional
        In "filter" mode, score threshold to use. 
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

```


### With AnnData/Scanpy
The [AnnData](https://anndata.readthedocs.io/) object stores single-cell data and metadata for [Scanpy](https://scanpy.readthedocs.io/), a popular Python workflow for single-cell analyses, as well as other Scanpy-compatible methods. BatchRefiner supports processing an embedding stored in the `obsm` field of an AnnData object: 
```
from batchrefiner import batchrefine_scanpy
import anndata as ad
#Input data
embed = ... #shape: (cells, dimensions)
batch_labels = ... #shape: (cells, )
#Create minimal AnnData
adata = ad.AnnData(X=None, obs={"batch":batch_labels}, obsm={"X_emb":embed}, shape=(cells, n_genes))

#Run BatchRefiner
def batchrefine_scanpy(adata, emb_key, batch_key="batch", br_key="X_BatchRefiner", score_key=None, copy=False, **kwargs):

batchrefine_scanpy(adata, "X_emb", batch_key="batch", br_key = "X_emb_scale_r2", mode="scale", metric="r2")
batchrefine_scanpy(adata, "X_emb", batch_key="batch", br_key="X_emb_filter_ilisi", mode="filter", metric="ilisi", n_proc=-1)
```
The complete set of parameters are very similar to `batchrefine`, with arrays swapped for keys in AnnData fields. 
```
>>> help(batchrefine_scanpy)
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
        BatchRefiner mode to use. Defaults to "scale"; "filter" or "scale" are also supported.
    metric : str | callable, optional
        Metric for scoring dimensions. Batch R^2 ("r2", default) and iLISI ("ilisi") are implemented using scib. 
        Otherwise, a user-supplied funciton is used. The function must take an embedding, an array of batch labels, 
        and optionally additional kwargs. It should return a higher score for columns with more batch signal.
    br_key : 
        Key to save BatchRefiner-modified embeddings in adata.obsm. 
        These will have shape (cells, dimensions) in "scale" mode or (cells, filter_dims) in "filter" mode.
    score_key : str, optional
        Key to save the scores, with shape (dimensions,), in adata.uns. Default is to not save scores.
    filter_dims : int, optional
        In "filter" mode, number of dimemsions to keep. 
    filter_thresh : float, optional
        In "filter" mode, score threshold to use. 
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

```

## Troubleshooting
### Using iLISI
If you encounter an error about `GLIBC` versions when using iLISI, this is because beacuse the pre-compiled scib package used by pip was built using a newer version of GLIBC than the version on your computer. To fix this, compile scib locally by running:
```
pip uninstall scib && pip install scib==1.1.7 --no-binary scib
```

If you encounter the following error:
```
AssertionError: daemonic processes are not allowed to have children
```
This is likely because both `n_proc` (number of processes used by BatchRefiner) and `n_cores` (number of processes used internally by scib's LISI) are greater than one. Unfortunately, parallelism is only supported at one level for iLISI, so one of those arguments must be 1 (their default). For better performance, it is suggested to set `n_proc=-1` and `n_cores=1`, versus setting `n_proc=1` and `n_cores=(# of CPUs)`. 

## References
Schäffer, D. E, Kang, H., Aksu, E. D., Edelman, D., Berger, B.: Ensemble learning significantly improves batch integration of scRNA-seq cell embeddings. *In preparation.*

Hie, B.L., Kim, S., Rando, T.A., Bryson, B., Berger, B.: Scanorama: integrating large and diverse single-cell
transcriptomic datasets. *Nat. Protoc.* **19**(8), 2283–2297 (Aug 2024)

