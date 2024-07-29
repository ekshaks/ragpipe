from typing import List
from .docnode import DocNode

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

def tokenize_remove_stopwords(text: str) -> List[str]:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize

    #nltk.download('punkt')
    #nltk.download('stopwords')

    tokens = word_tokenize(text.lower())  
    filtered_tokens = [token for token in tokens if token not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]  # Stemming

    return stemmed_tokens

#https://github.com/dorianbrown/rank_bm25
from rank_bm25 import BM25Okapi
import numpy as np

class BM25:
    def __init__(self, doc_list) -> None: #expect: doc.get_content() defined
        self.doc_list = doc_list
        self._corpus = [self._tokenizer(doc) for doc in doc_list]
        self.bm25 = BM25Okapi(self._corpus)
        
    def _tokenizer(self, text):
        return tokenize_remove_stopwords(text)
    
    def retrieve(self, query, limit=10):
        from .docnode import ScoreNode
        if isinstance(query, str):
            tokenized_query = self._tokenizer(query)
        else:
            assert isinstance(query, list)
            tokenized_query = query
        scores = self.bm25.get_scores(tokenized_query)
        top_n = np.argsort(scores)[::-1][:limit]
        node_scores = [ScoreNode(li_node=self.doc_list[i], score=scores[i], doc_path_index=i) 
                       for i in top_n]
        #doc_scores = zip(self.doc_list, scores)
        #node_scores = [ScoreNode(li_node=x[0], score=x[1], doc_path_index=j) for j, x in 
        #               enumerate(doc_scores)]
        nodes_scores_sorted = sorted(node_scores, key=lambda x: x.score or 0.0, reverse=True)
        return nodes_scores_sorted[: limit]
    


def qD_sparse_similarity(doc_embeddings=None, query_embedding=None):
    #both doc_embeddings and query_embedding are sparse 
    assert doc_embeddings is not None and query_embedding is not None
    import torch
    import torch.nn.functional as F

    #scores = [ torch.matmul(query_embedding, demb) for demb in doc_embeddings] #dotprod for sparse on CPU?
    doc_embeddings = torch.stack([doc.to_dense() for doc in doc_embeddings]) #(d,)* -> (n, d)
    scores = F.cosine_similarity(query_embedding.to_dense().unsqueeze(0), doc_embeddings).tolist()
    return scores


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


from pydantic import BaseModel
from typing import Any
from .config import EncoderConfig

class Encoder(BaseModel):
    name: str
    mo_loader: Any
    rep_type: str #= 'single_vector', 'multi_vector', 'object'
    config: EncoderConfig
    _model: Any | None = None

    def get_model(self):
        if self._model is None:
            self._model = self.mo_loader()
        return self._model
    
    def encode(self, docs, is_query=False) : #for RPIndex
        raise NotImplementedError('call the encode function for derived Encoder.')


class BM25Encoder(Encoder):
    def encode(self, docs, is_query=False):
        #print(f'BM25Enc.encode: is_query= {is_query} ')
        reps = docs if is_query else BM25(docs)
        return reps

    @classmethod
    def from_config(cls, config):
        name = config.name
        return BM25Encoder(name=name, mo_loader=None, config=config,
                                rep_type='single_vector') 


class LLMEncoder(Encoder):
    @classmethod
    def from_config(cls, config): 
        return LLMEncoder(name=config.name, mo_loader='', rep_type='object', config=config)
    
class LlamaIndexEncoder(Encoder):

    @classmethod
    def from_config(cls, config): 
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
        from llama_index.embeddings.clip import ClipEmbedding
        name = config.name
        if 'ViT' not in name:
            model_loader = lambda: FastEmbedEmbedding(model_name=name)
        else:
            model_loader = lambda: ClipEmbedding(model_name=name) #"ViT-B/32")

        return LlamaIndexEncoder(name=name, mo_loader=model_loader, config=config,
                                rep_type='single_vector')
    
    def encode(self, docs, is_query=False) -> 'List[torch.Tensor]': #for RPIndex
        raise ValueError('Cannot be called as LLamaIndex encodes under the hood.')

class FastEmbedEncoder(Encoder):
    @classmethod
    def from_config(cls, config): 
        dense_models = ["BAAI/bge-small-en-v1.5"]
        sparse_models = ["prithivida/Splade_PP_en_v1"]
        name = config.name
        match name:
            case dm if dm in dense_models:
                model_loader = lambda: TextEmbedding(model_name=name)
            case sm if sm in sparse_models:
                model_loader = lambda: SparseTextEmbedding(model_name=name)
            case _:
                raise ValueError(f"FastEmbedEncoder: unknown model {name}")
        return FastEmbedEncoder(name=name, mo_loader=model_loader, config=config,
                                rep_type='single_vector')

    def encode(self, docs, is_query=False) -> 'List[torch.Tensor]': #for RPIndex
        encode_fn = self.get_model().embed #fastembed
        printd(2, 'fastembed encode ..')
        doc_embeddings = [np_to_torch(x) for x in encode_fn(docs, show_progress=True)]
        printd(2, 'fastembed encode over..')

        return doc_embeddings
    
    def get_similarity_fn(self):
        sim_fn = None
        printd(2, f'get_similarity_fn: {self.name}')
        match self.name:
            case 'prithivida/Splade_PP_en_v1':
                sim_fn = qD_sparse_similarity
            case _:
                sim_fn = None
        return sim_fn

class ColbertEncoder(Encoder):
    @classmethod
    def from_config(cls, config): 
        name = config.name
        match name:
            case "colbert-ir/colbertv2.0":
                model_loader = lambda: Colbert(model=name, tokenizer=name)
            case "jinaai/jina-colbert-v1-en":
                model_loader = lambda: Colbert(model=name, tokenizer=name, max_length=8192)
            case _:
                raise ValueError(f"ColbertEncoder: unknown model {name}")
        return ColbertEncoder(name=name, mo_loader=model_loader, config=config,
                                rep_type='multi_vector')        
    
    def encode(self, docs, is_query=False) -> 'List[torch.Tensor]': #for RPIndex
        encode_fn = self.get_model().get_text_embedding 
        doc_embeddings=[encode_fn(doc) for doc in tqdm(docs)]
        return doc_embeddings

    def get_similarity_fn(self):
        sim_fn = None
        printd(2, f'get_similarity_fn: {self.name}')
        match self.name:
            case "colbert-ir/colbertv2.0":
                sim_fn = self.get_model().compute_similarity_embeddings #TODO: make static member of Colbert
            case "jinaai/jina-colbert-v1-en":
                sim_fn = self.get_model().compute_relevance_scores
            case _:
                sim_fn = None
        return sim_fn
    
class BGE_M3:
    #https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3
    #https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py
    pass

class MixbreadEncoder(Encoder):
    #https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1

    @classmethod
    def from_config(cls, name, config): #name = "mixedbread-ai/mxbai-embed-large-v1"
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            print('The mixbread embeddings need sentence_transformers library installed.')

        model_loader = lambda: SentenceTransformer(name, truncate_dim=config.size)
        return cls(name='mixbread', mo_loader=model_loader, rep_type = 'single_vector', config=config)
    

    def _transform_query(query: str) -> str:
        """ For retrieval, add the prompt for query (not for documents).
        """
        return f'Represent this sentence for searching relevant passages: {query}'

        
    def encode(self, docs, size=1024, dtype='float32', is_query=False):
        from sentence_transformers.quantization import quantize_embeddings
        dtype2precision = { #TODO: check, fix this map
                    'float32': 'float32', 'float16': 'float16', 'bfloat16': 'bfloat16',
                    'binary': 'ubinary'
        }
        embeddings = self.get_model().encode(docs) #TODO: check size?
        dtype = self.config.dtype
        embeddings = quantize_embeddings(embeddings, precision=dtype2precision[dtype])
    
        return embeddings
    
    def get_similarity_fn(self):
        assert self.name == "mixedbread-ai/mxbai-embed-large-v1"
        from sentence_transformers.util import cos_sim
        return cos_sim
    

EncoderPool = {}

def get_encoder(econfig, **kwargs):
    doc_leaf_type = kwargs.get('doc_leaf_type', 'raw')
    if doc_leaf_type == 'llamaindex':
        return LlamaIndexEncoder.from_config(econfig)

    fastembed_models = ["BAAI/bge-small-en-v1.5",
                        "prithivida/Splade_PP_en_v1"] #read from file
    colbert_models = [
        "colbert-ir/colbertv2.0",
        "jinaai/jina-colbert-v1-en"
    ]

    #TODO: add config hash + encoder name
    encoder = EncoderPool.get(econfig, None) 
    if encoder is not None:
        return encoder
    
    printd(3, f'get_encoder: {econfig}')
    
    match econfig.name:
        case 'bm25':
            encoder = BM25Encoder.from_config(econfig)
        case fe if fe in fastembed_models:
            encoder = FastEmbedEncoder.from_config(econfig)
        case ce if ce in colbert_models:
            encoder = ColbertEncoder.from_config(econfig)
        case mxbai if 'mxbai' in mxbai:
            assert False, f"not tested yet : {econfig}"
            encoder = MixbreadEncoder.from_config(name, config)
        case llme if 'llm' in llme:
            encoder = LLMEncoder.from_config(econfig)
        case _:
            raise ValueError(f'Not handled encoder : {econfig}')
    
    EncoderPool[econfig] = encoder 
    return encoder
