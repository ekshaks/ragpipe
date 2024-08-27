
from typing import List

from fastembed import SparseEmbedding
from .common import printd

#TODO!
'''
def qD_binary_similarity(doc_embeddings, query_embedding):
    # compute hamming distance
    pass
'''

def qD_cosine_similarity(doc_embeddings: 'list(d,)'=None, query_embedding: '(d,)'=None):
    import torch.nn.functional as F
    from torch import stack

    assert doc_embeddings is not None and query_embedding is not None
    doc_embeddings = stack(doc_embeddings) #(d,)* -> (n, d)
    scores = F.cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings).tolist()
    #scores = [ F.cosine_similarity(query_embedding, demb, dim=0) for demb in doc_embeddings]
    return scores


def exact_nn(doc_embeddings, doc_paths, rep_query, similarity_fn=None, limit=None) -> 'List[Result]':
    '''Exact nearest neighbors of rep_query and doc_embeddings
    doc_embeddings: list of single- or multi-vector embeddings
    doc_paths: path to each doc part 
    req_query: single vector embedding
    TODO: return SearchResults(doc_path=, score=) -> sort() -> ScoreNode.fromResults(..)
    '''
    if similarity_fn is None:
        similarity_fn = qD_cosine_similarity
    #printd(3, doc_embeddings)
    #printd(3, rep_query)
    #printd(3, f'exact_nn shapes: doc = {doc_embeddings[0].size()}, query = {rep_query.size()}')
    scores = similarity_fn(doc_embeddings=doc_embeddings, query_embedding=rep_query)
    printd(3, f'exact_nn: scores = {scores}')
    results = [dict(doc_path=doc_path, score=score) for doc_path, score in zip(doc_paths, scores)]
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:limit]
    return results


def qD_sparse_similarity(doc_embeddings=None, query_embedding=None):
    #both doc_embeddings and query_embedding are sparse 
    assert doc_embeddings is not None and query_embedding is not None
    import torch
    import torch.nn.functional as F

    #scores = [ torch.matmul(query_embedding, demb) for demb in doc_embeddings] #dotprod for sparse on CPU?
    doc_embeddings = torch.stack([doc.to_dense() for doc in doc_embeddings]) #(d,)* -> (n, d)
    scores = F.cosine_similarity(query_embedding.to_dense().unsqueeze(0), doc_embeddings).tolist()
    return scores


def np_to_torch(x, vocab_size=30522):
    #tokenizer(prithivida/Splade_PP_en_v1).vocab_size (30522)

    import torch
    #is_sparse = scipy.sparse.issparse(x)
    #printd(2, f'np_to_torch: x is sparse {is_sparse}')
    try:
        dense = torch.Tensor(x)
        return dense
    except Exception as e:
        assert isinstance(x, SparseEmbedding), f'type of x {type(x)}'
        #https://stackoverflow.com/questions/63076260/how-to-create-a-1d-sparse-tensors-from-given-list-of-indices-and-values
        i = torch.LongTensor(x.indices).unsqueeze(0)
        v = torch.FloatTensor(x.values)
        #print(i)
        #print(v)
        shape = torch.Size((vocab_size,)) 
        sparse = torch.sparse_coo_tensor(i, v, shape)
        return sparse
