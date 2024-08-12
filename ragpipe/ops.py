
from typing import List
from .common import printd


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
    printd(3, f'exact_nn shapes: doc = {doc_embeddings[0].size()}, query = {rep_query.size()}')
    scores = similarity_fn(doc_embeddings=doc_embeddings, query_embedding=rep_query)
    printd(3, f'exact_nn: scores = {scores}')
    results = [dict(doc_path=doc_path, score=score) for doc_path, score in zip(doc_paths, scores)]
    results = sorted(results, key=lambda x: x['score'], reverse=True)[:limit]
    return results
