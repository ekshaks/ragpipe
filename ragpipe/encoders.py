from typing import List
from .docnode import DocNode

from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from fastembed import SparseTextEmbedding, SparseEmbedding, TextEmbedding

from .colbert import Colbert
from .common import DotDict, printd

from tqdm import tqdm

def tokenize_remove_stopwords(text: str) -> List[str]:
    from nltk.stem import PorterStemmer
    # lowercase and stem words
    text = text.lower()
    stemmer = PorterStemmer()
    words = text.split(' ') 
    return [stemmer.stem(word) for word in words]

#https://github.com/dorianbrown/rank_bm25
from rank_bm25 import BM25Okapi


class BM25:
    def __init__(self, doc_list) -> None: #expect: doc.get_content() defined
        self.doc_list = doc_list
        self._corpus = [self._tokenizer(doc.get_content()) for doc in doc_list]
        self.bm25 = BM25Okapi(self._corpus)
        
    def _tokenizer(self, text):
        return tokenize_remove_stopwords(text)
    
    def retrieve(self, query, top_k=5):
        tokenized_query = self._tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)
        node_scores = [DocNode(li_node=x[0], score=x[1]) for x in zip(self.doc_list, scores)]
        nodes_scores_sorted = sorted(node_scores, key=lambda x: x.score or 0.0, reverse=True)
        return nodes_scores_sorted[: top_k]
    

#https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1
def get_mixbread_encode_fn(size=1024, dtype='float32'):
    '''
    TODO: combine this with encode_fn, get_similarity_fn below
    return dict(encode_fn=..., similarity_fn=..., transform_query_fn=..., index_type=..., rep_type=...)

    '''
    def transform_query(query: str) -> str:
        """ For retrieval, add the prompt for query (not for documents).
        """
        return f'Represent this sentence for searching relevant passages: {query}'

    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import cos_sim
        from sentence_transformers.quantization import quantize_embeddings
    except Exception as e:
        print('The mixbread embeddings need sentence_transformers library installed.')

    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=size)
    dtype2precision = { #TODO: check, fix this map
        'float32': 'float32', 'float16': 'float16', 'bfloat16': 'bfloat16',
        'binary': 'ubinary'
    }
    def encode_fn(docs):
        embeddings = model.encode(docs)
        embeddings = quantize_embeddings(embeddings, precision=dtype2precision[dtype])
    return encode_fn

def get_encoder_index_reptype (encoder_name, doc_leaf_type='raw', **kwargs):
    #doc_leaf_type - is it plain text/image or wrapped in LI node
    index_type = 'rpindex'
    rep_type = 'single_vector'
    match encoder_name:
        case "BAAI/bge-small-en-v1.5":
            if doc_leaf_type == 'llamaindex':
                embed_model = lambda: FastEmbedEmbedding(model_name=encoder_name)
                index_type = 'llamaindex'
            else:
                embed_model = lambda: TextEmbedding(model_name=encoder_name)

        case "prithivida/Splade_PP_en_v1":
            if doc_leaf_type == 'llamaindex':
                embed_model = lambda: FastEmbedEmbedding(model_name=encoder_name)
                index_type = 'llamaindex'
            else:
                embed_model = lambda: SparseTextEmbedding(model_name=encoder_name)

        case "colbert-ir/colbertv2.0":
            embed_model = lambda: Colbert(model=encoder_name, tokenizer=encoder_name)
            rep_type = 'multi_vector'
        case "jinaai/jina-colbert-v1-en":
            embed_model = lambda: Colbert(model=encoder_name, tokenizer=encoder_name, max_length=8192)
            rep_type = 'multi_vector'

        case "mixedbread-ai/mxbai-embed-large-v1":
            embed_model = get_mixbread_encode_fn(size=kwargs.get('dimensions', 1024), 
                                                      dtype=kwargs.get('dtype', 'float32'))
            rep_type = 'single_vector'
        
        case "ViT-B/32":
            if doc_leaf_type == 'llamaindex':
                embed_model = lambda: ClipEmbedding(model_name="ViT-B/32")
                index_type = 'llamaindex'
            else:
                raise NotImplementedError()
        
        case _:
            embed_model = lambda: None
            #raise ValueError(f"unknown encoder: {encoder_name}")
            
    return DotDict(index_type=index_type, encoder_model_loader=embed_model, rep_type=rep_type)

def qD_sparse_similarity(doc_embeddings=None, query_embedding=None):
    #both doc_embeddings and query_embedding are sparse 
    assert doc_embeddings is not None and query_embedding is not None
    import torch
    import torch.nn.functional as F

    #scores = [ torch.matmul(query_embedding, demb) for demb in doc_embeddings] #dotprod for sparse on CPU?
    doc_embeddings = torch.stack([doc.to_dense() for doc in doc_embeddings]) #(d,)* -> (n, d)
    scores = F.cosine_similarity(query_embedding.to_dense().unsqueeze(0), doc_embeddings).tolist()
    return scores

def get_similarity_fn(encoder_name, encoder_model):
    sim_fn = None
    match encoder_name:
        case "colbert-ir/colbertv2.0":
            sim_fn = encoder_model.compute_similarity_embeddings
        case 'prithivida/Splade_PP_en_v1':
            sim_fn = qD_sparse_similarity
        case "mixedbread-ai/mxbai-embed-large-v1":
            from sentence_transformers.util import cos_sim
            sim_fn = cos_sim
        case _:
            sim_fn = None
    return sim_fn


def np_to_torch(x):
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
        shape = torch.Size((30522,)) #tokenizer(prithivida/Splade_PP_en_v1).vocab_size
        sparse = torch.sparse_coo_tensor(i, v, shape)
        return sparse

def encode_fn(encoder_model, docs) -> 'List[torch.Tensor]': #encode for RPIndex
    try:
        encode_fn = encoder_model.get_text_embedding #colbert
        doc_embeddings=[encode_fn(doc) for doc in tqdm(docs)]
    except Exception as e:
        encode_fn = encoder_model.embed #fastembed
        printd(2, 'fastembed encode ..')
        doc_embeddings = [np_to_torch(x) for x in encode_fn(docs, show_progress=True)]
        printd(2, 'fastembed encode over..')

    return doc_embeddings