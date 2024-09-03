from typing import List
from tqdm import tqdm


from fastembed import SparseTextEmbedding, TextEmbedding

from .common import DotDict, printd, load_func
from .docnode import DocNode
from .ops import np_to_torch, qD_sparse_similarity, qD_cosine_similarity
from .colbert import Colbert

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
        raise NotImplementedError(f'Please implement the encode function for derived Encoder class {self.__class__}.')

class PassThroughEncoder(Encoder):
    def encode(self, docs, is_query=False):
        return docs
    @classmethod
    def from_config(cls, config):
        return PassThroughEncoder(name=config.name, mo_loader=None, config=config, rep_type='object')


'''
class BM25Encoder(Encoder):
    def encode(self, docs, is_query=False):
        #print(f'BM25Enc.encode: is_query= {is_query} ')
        reps = docs if is_query else RankBM25Index(docs)
        return reps

    @classmethod
    def from_config(cls, config):
        name = config.name
        return BM25Encoder(name=name, mo_loader=None, config=config,
                                rep_type='single_vector') 
'''


class LLMEncoder(Encoder):
    def encode(self, docs, is_query=False):
        from . import llm_bridge
        prompt = self.config.prompt
        return llm_bridge.transform(docs, self.config.name, prompt=prompt, is_query=is_query)
    
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
    def from_config(cls, config: EncoderConfig): 
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

    def prefix_query_instruction(self, docs):
        qi = self.config.query_instruction
        if qi is not None:
            docs = [ f'{qi} {d}' for d in docs]
        return docs

    def encode(self, docs, is_query=False) -> 'List[torch.Tensor]': #for RPIndex
        if is_query: docs = self.prefix_query_instruction(docs)
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
                sim_fn = qD_cosine_similarity
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
        #case 'bm25': encoder = BM25Encoder.from_config(econfig)
        case fe if fe in fastembed_models:
            encoder = FastEmbedEncoder.from_config(econfig)
        case ce if ce in colbert_models:
            encoder = ColbertEncoder.from_config(econfig)
        case mxbai if 'mxbai' in mxbai:
            assert False, f"not tested yet : {econfig}"
            encoder = MixbreadEncoder.from_config(name, config)
        case llme if 'llm' in llme:
            encoder = LLMEncoder.from_config(econfig)
        case pth if 'passthrough' in pth:
            encoder = PassThroughEncoder.from_config(econfig)
        case _:
            if not econfig.with_index and econfig.module is not None:
                encoder = load_func(econfig.module).from_config(econfig)
            else:
                printd(1, f'Encoder undefined: {econfig}. Assuming nop encoder (passthrough).')
                #TODO: expect 'encoder_module': Passthrough, else ValueError
                encoder = PassThroughEncoder.from_config(econfig)
    
    EncoderPool[econfig] = encoder 
    return encoder

def get_encoder_reptype(econfig: EncoderConfig):
    #TODO: fix inaccurate reptypes
    if econfig.shape is not None:
        return econfig.shape.rep_type

    colbert_models = [
        "colbert-ir/colbertv2.0",
        "jinaai/jina-colbert-v1-en"
    ]
    
    printd(3, f'get_encoder_reptype: {econfig}')
    rep_type = 'single_vector'
    match econfig.name:
        case ce if ce in colbert_models:
            rep_type = 'multi_vector'
    
    return rep_type