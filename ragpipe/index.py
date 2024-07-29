
from llama_index.core import VectorStoreIndex
from .common import DEFAULT_LIMIT, printd
from .db import Storage, StorageConfig, exact_nn


from pydantic import BaseModel


from typing import List, Optional

from .docnode import ScoreNode
from .encoders import get_encoder
from .config import EncoderConfig

class IndexConfig(BaseModel):
    index_type: str
    encoder_config: EncoderConfig
    doc_paths: List[str]
    storage_config: Optional[StorageConfig]

    '''StorageConfig
    collection: str
    path: str = '/tmp/ragpipe'
    dbname: str = 'chromadb'
    '''


class RPIndex(): #rag pipe index
    def __init__(self, encoder,
                 storage_config: StorageConfig =None):
        #self.encoder_name = encoder_name
        #self.encoder_model = encoder_model
        self.encoder = encoder
        self.storage_config = storage_config

        #for in mem storage
        self.doc_paths = [] #update when add
        self.doc_embeddings = [] #
        self.is_query = False #update when 'add'
        self.storage = None

    def get_index_config(self):
        return IndexConfig(index_type='rpindex', storage_config=self.storage_config,
                           encoder_config=self.encoder.config, doc_paths=self.doc_paths)

    def get_storage(self):
        if self.storage is None:
            assert self.storage_config is not None
            self.storage = Storage(self.storage_config)
        return self.storage

    @classmethod
    def from_index_config(cls, ic: IndexConfig):
        encoder = get_encoder(ic.encoder_config)
        #return cls(ic.encoder_name, encoder_model, ic.storage_config)
        return cls(encoder, ic.storage_config)

    def add(self, docs, doc_paths, is_query=False, docs_already_encoded=False):
        self.doc_paths.extend(doc_paths)

        self.is_query = is_query
        if not docs_already_encoded:
            #doc_embeddings = encode_fn(self.encoder_model, docs)
            doc_embeddings = self.encoder.encode(docs, is_query=is_query)

        else:
            doc_embeddings = docs

        if self.storage_config is not None:
            storage = self.get_storage()
            storage.add(doc_embeddings, doc_paths)
        else:
            self.doc_embeddings = doc_embeddings


    def get_query_rep(self):
        assert self.is_query, f'cant get rep from non-query index: {self.doc_paths}'
        return self.doc_embeddings[0]

    def retrieve_in_mem(self, rep_query, similarity_fn=None, limit=None):
        return exact_nn(self.doc_embeddings, self.doc_paths, rep_query,
                        similarity_fn=similarity_fn,
                        limit=limit)

    def retrieve(self, rep_query, limit=DEFAULT_LIMIT):
        if self.encoder.name == 'bm25':
            bm = self.doc_embeddings
            doc_nodes = bm.retrieve(rep_query, limit=limit)
            for d in doc_nodes:
                d.doc_path = self.doc_paths[d.doc_path_index]
            return doc_nodes
        
        similarity_fn = self.encoder.get_similarity_fn()

        doc_nodes: List[ScoreNode]
        if self.storage_config is None:
            assert similarity_fn is not None
            doc_nodes = self.retrieve_in_mem(rep_query, similarity_fn=similarity_fn, limit=limit)
        else:
            storage = self.get_storage()
            doc_nodes = storage.retrieve(rep_query, 
                                        similarity_fn=similarity_fn,
                                        limit=limit)
        return doc_nodes


class VectorStoreIndexPath(VectorStoreIndex):
    def __init__(self, index_config:IndexConfig, vsi: VectorStoreIndex = None):
        self.index_config = index_config
        self.vsi = vsi

    def get_vector_store_index(self): return self.vsi

    @classmethod
    def from_docs(cls, docs, ic: IndexConfig, encoder_model):
        item_type = type(docs[0]).__name__
        if ic.storage_config is not None:
            storage = Storage(ic.storage_config)
            ctx = storage.get_LI_context()
            storage_context = ctx.storage_context
        else:
            storage_context = None


        if 'TextNode' in item_type:
            vsi = VectorStoreIndex(docs, embed_model=encoder_model,
                                   storage_context=storage_context, show_progress=True)
        else:
            vsi = VectorStoreIndex.from_documents(docs, embed_model=encoder_model,
                        storage_context=storage_context, show_progress=True)


        return cls(vsi=vsi, index_config=ic)

    @classmethod
    def from_index_config(cls, ic: IndexConfig):
        storage = Storage(ic.storage_config)
        ctx = storage.get_LI_context()
        #encoder_model = get_encoder_index(ic.encoder.name).encoder_model

        vsi = VectorStoreIndex.from_vector_store(
                ctx.vector_store, storage_context=ctx.storage_context
            )
        return cls(vsi=vsi, index_config=ic)

    def get_index_config(self):
        return self.index_config


class ObjectIndex(): # list of objects
    def __init__(self, reps, paths, is_query=False, docs_already_encoded=False):
        self.reps, self.paths = reps, paths
        self.is_query = is_query
    def get_query_rep(self):
        assert self.is_query, 'Index does not store query rep'
        return self.reps[0]
    def items(self):
        '''Returns a list of tuples (rep, path) contained in the index
        '''
        return zip(self.reps, self.paths)
    def get_index_config(self):
        #return IndexConfig(index_type='objectindex', storage_config=self.storage_config, 
        #                   encoder_name=self.encoder.name, doc_paths=self.doc_paths)
        raise NotImplementedError()
    @classmethod
    def from_index_config(cls, ic: IndexConfig):
        raise NotImplementedError()
    def __str__(self):
        return(f'\nObjectIndex: \n reps = {self.reps} \n paths = {self.paths}\n')


class IndexManager:
    def __init__(self, path='/tmp/ragpipe/'):
        import diskcache as dc
        self.path = path
        self.cache = dc.Cache(path)

    def get_key(self, fpath, repname, encoder_name):
        return f'{fpath}#{repname}#{encoder_name}'

    def add(self, fpath, repname, encoder_name, index):
        #try:
        index_config: IndexConfig = index.get_index_config()
        key = self.get_key(fpath, repname, encoder_name)
        self.cache[key.encode('utf-8')] = index_config
        printd(1, f'adding {key} for index {type(index)}')
        '''
        except Exception as e:
            import traceback
            printd(1, f'not adding {fpath} for index {type(index)}.\n Exception: {e}')
            traceback.print_exc()
        '''


    def get_config(self, fpath, repname, encoder_name) -> IndexConfig:
        key = self.get_key(fpath, repname, encoder_name)
        try:
            val = self.cache[key.encode('utf-8')]
        except:
            val = None
        #printd(2, f'IM::loaded config -> {val}')
        return val