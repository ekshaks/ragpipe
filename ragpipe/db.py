from typing import Union, List
from pathlib import Path
from .common import DEFAULT_LIMIT

import chromadb
from pydantic import BaseModel
from safetensors import safe_open as st_safe_open
from safetensors.torch import save_file as st_save_file

from .common import printd, DotDict
from .docnode import ScoreNode #can we remove this dep?

def qD_cosine_similarity(doc_embeddings: 'list(d,)'=None, query_embedding: '(d,)'=None):
    import torch.nn.functional as F
    from torch import stack

    assert doc_embeddings is not None and query_embedding is not None
    doc_embeddings = stack(doc_embeddings) #(d,)* -> (n, d)
    scores = F.cosine_similarity(query_embedding.unsqueeze(0), doc_embeddings).tolist()
    #scores = [ F.cosine_similarity(query_embedding, demb, dim=0) for demb in doc_embeddings]
    return scores



def exact_nn(doc_embeddings, doc_paths, rep_query, similarity_fn=None, limit=None) -> List[ScoreNode]:
    '''Exact nearest neighbors of rep_query and doc_embeddings
    doc_embeddings: list of single- or multi-vector embeddings
    doc_paths: path to each doc part 
    req_query: single vector embedding
    TODO: optimize!
    '''
    if similarity_fn is None:
        similarity_fn = qD_cosine_similarity
    printd(3, f'exact_nn shapes: doc = {doc_embeddings[0].size()}, query = {rep_query.size()}')
    scores = similarity_fn(doc_embeddings=doc_embeddings, query_embedding=rep_query)
    printd(3, f'exact_nn: scores = {scores}')
    doc_nodes = [ScoreNode(doc_path=doc_path, is_ref=True, score=score) 
        for doc_path, score in zip(doc_paths, scores)]
    doc_nodes = sorted(doc_nodes, key=lambda x: x.score, reverse=True)[:limit]
    return doc_nodes

class TensorCollection:
    def __init__(self, dbpath, collection):
        self.dbpath = dbpath
        self.collection = collection
        self.file_path = Path(dbpath) / f'{collection}.safetensors'
    
    @staticmethod
    def flatten(key, tlist):
        flat_dict = {}
        if not isinstance(tlist, list): #passthrough single vectors
            #print('flatten: ', key, ' ', tlist.size())
            flat_dict[key] = tlist
        else:
            for i in range(len(tlist)): #separate keys for multi vectors
                flat_dict[f'{key}~{i}'] = tlist[i]
        return flat_dict
    
    @staticmethod
    def unflatten(tdict):
        from collections import defaultdict
        out_dict = defaultdict(list)

        for key, value in tdict.items():
            if '~' in key:
                key_prefix, pos = key.split('~')
                out_dict[key_prefix].insert(int(pos), value)
            else:
                out_dict[key] = value
        doc_paths = [key for key in out_dict]
        doc_embeddings = [out_dict[key] for key in out_dict]
        return doc_embeddings, doc_paths
    
    def add(self, reps, paths):
        tensor_dict = {}
        for path, rep in zip(paths, reps):
            upd_dict = TensorCollection.flatten(path, rep)
            tensor_dict.update(upd_dict)
        st_save_file(tensor_dict, self.file_path)

    def retrieve(self, rep_query, similarity_fn=None, limit=DEFAULT_LIMIT):
        tensor_dict = {}
        with st_safe_open(self.file_path, framework="pt", device="cpu") as f:
            for path in f.keys():
                tensor_dict[path] = f.get_tensor(path)
        printd(2, f'TensorColl.retrieve: {list(tensor_dict.keys())}')
        doc_embeddings, doc_paths = TensorCollection.unflatten(tensor_dict)
        docnodes = exact_nn(doc_embeddings, doc_paths, rep_query, similarity_fn=similarity_fn, limit=limit)
        return docnodes

        

class StorageConfig(BaseModel):
    collection_name: str
    path: str = '/tmp/ragpipe'
    rep_type: str = 'single_vector'
    dbname: str = 'chromadb'

    def __init__(self, **kwargs):
    #def __post_model_init__(self):
        if kwargs['rep_type'] == 'multi_vector':
            kwargs['dbname'] = 'tensordb'
        super().__init__(**kwargs)
        Path(self.path).mkdir(parents=True, exist_ok=True)

        #assert False, f'{self.dbname}, {self.rep_type}'




class Storage:
    def __init__(self, config: StorageConfig) -> None:
        self.C = config
        print(f'init storage: {config}')

        match config.dbname:
            case 'chromadb':
                self.client = chromadb.PersistentClient(path=self.C.path)
                self.collection = self.client.get_or_create_collection(self.C.collection_name)
            case 'tensordb':
                self.collection = TensorCollection(dbpath=self.C.path, collection=self.C.collection_name)
            case _:
                raise NotImplementedError()
        # get items from a collection
        #collection.get()
        # get first 5 items from a collection
        #collection.peek()
    
    def get_LI_context(self):
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import StorageContext
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return DotDict(storage_context=storage_context, vector_store=vector_store)
    

    def add(self, reps, paths):
        printd(2, f'adding to storage {self.C.dbname} - paths: {paths}')
        match self.C.dbname:
            case 'chromadb':
                self.collection.add(
                    embeddings=[rep.tolist() for rep in reps], #expects a list of list
                    metadatas=[dict(path=p) for p in paths],
                    ids=paths
                )
            case 'tensordb':
                self.collection.add(reps, paths)

    def retrieve_chromadb(self, rep_query, limit=DEFAULT_LIMIT):
        # do nearest neighbor search to find similar embeddings or documents, supports filtering
        results = self.collection.query(
            query_embeddings=[rep_query],
            n_results=limit,
           # where={"style": "style2"}
        )

        doc_idx, query_idx = 0, 0
        #category = results['metadatas'][query_idx][doc_idx][topic_field]
        #distance = results['distances'][query_idx][doc_idx]

        docnodes = []
        metadatas = results['metadatas'][query_idx]
        distances = results['distances'][query_idx]
        idx_len = min(limit, len(metadatas))
        for doc_idx in range(idx_len):
            docnodes.append(ScoreNode(doc_path=metadatas[doc_idx]['path'], score = 2-distances[doc_idx]))
        return docnodes
    
    def retrieve_tensordb(self, rep_query, similarity_fn=None, limit=DEFAULT_LIMIT):
        return self.collection.retrieve(rep_query, similarity_fn=similarity_fn, limit=limit)
    
    def retrieve(self, rep_query, similarity_fn=None, limit=DEFAULT_LIMIT):
        printd(2, f'retrieving from storage {self.C.dbname}')
        match self.C.dbname:
            case 'chromadb':
                docnodes = self.retrieve_chromadb(rep_query.tolist(), limit=limit)
            case 'tensordb':
                docnodes = self.retrieve_tensordb(rep_query, similarity_fn=similarity_fn, limit=limit)
            case _:
                raise NotImplementedError(f'retrieve: {self.C.dbname}')
        return docnodes

