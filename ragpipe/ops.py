

def qD_cosine_similarity(doc_embeddings: 'list(d,)'=None, query_embedding: '(d,)'=None):
    import torch.nn.functional as F
    from torch import stack

    assert doc_embeddings is not None and query_embedding is not None
    doc_embeddings = stack(doc_embeddings) #(d,)* -> (n, d)
    scores = F.cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings).tolist()
    #scores = [ F.cosine_similarity(query_embedding, demb, dim=0) for demb in doc_embeddings]
    return scores
