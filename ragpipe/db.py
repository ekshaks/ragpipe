from typing import Union, Optional
from pathlib import Path
import uuid

#import chromadb
from pydantic import BaseModel

from .common import printd, DotDict, DEFAULT_LIMIT
from .ops import exact_nn

from .docnode import ScoreNode #can we remove this dep?

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
        from safetensors.torch import save_file as st_save_file

        tensor_dict = {}
        for path, rep in zip(paths, reps):
            upd_dict = TensorCollection.flatten(path, rep)
            tensor_dict.update(upd_dict)
        st_save_file(tensor_dict, self.file_path)

    def retrieve(self, rep_query, similarity_fn=None, limit=DEFAULT_LIMIT):
        from safetensors import safe_open as st_safe_open

        tensor_dict = {}
        with st_safe_open(self.file_path, framework="pt", device="cpu") as f:
            for path in f.keys():
                tensor_dict[path] = f.get_tensor(path)
        printd(2, f'TensorColl.retrieve: {list(tensor_dict.keys())}')
        doc_embeddings, doc_paths = TensorCollection.unflatten(tensor_dict)
        results = exact_nn(doc_embeddings, doc_paths, rep_query, similarity_fn=similarity_fn, limit=limit)
        #results: '(doc_path | score)*' #sorted
        docnodes = [ScoreNode(doc_path=r['doc_path'], is_ref=True, score=r['score']) for r in results]
        return docnodes

        
from .config import DBConfig

class StorageConfig(BaseModel):
    db_props: DBConfig
    collection_name: str
    rep_type: str = 'single_vector'
    size: Optional[int] = 384 #TODO: move this to rep_type config

    def get_dimension(self):
        return self.size

    @classmethod
    def from_kwargs(cls, **kwargs):
        collection_name = kwargs.pop('collection_name')
        rep_type = kwargs.pop('rep_type')
        db_props = kwargs.pop('db_props')

        if isinstance(db_props, bool): #resolve store=True
            dbs = kwargs.pop('dbs')
            if rep_type == 'single_vector':
                db_props = dbs['__default_single_vector__']
            elif rep_type == 'multi_vector':
                db_props = dbs['__default_multi_vector__']
            else:
                raise ValueError(f'Invalid rep_type {rep_type}')
        else:
            assert isinstance(db_props, DBConfig), f'unknown db_props type {db_props}'
        
        sc = cls(collection_name=collection_name, rep_type=rep_type, db_props=db_props)
        return sc

    
    def __init__(self, **kwargs):
        #if kwargs['rep_type'] == 'multi_vector':
        #    kwargs['dbname'] = 'tensordb'
        super().__init__(**kwargs)
        Path(self.db_props.path).mkdir(parents=True, exist_ok=True)


class _QdrantDB:
    '''
    collection_name: str
    client: Any
    dimension: int 
    metric: str = 'cosine'

    ref: https://colab.research.google.com/drive/1Bz8RSVHwnNDaNtDwotfPj0w7AYzsdXZ-?usp=sharing#scrollTo=UHpR7nNreM1B
    '''

    def __init__(self, collection_name, dimension, path=None, distance='cosine', recreate=False):
        import qdrant_client as qc
        from qdrant_client.models import Distance, VectorParams

        self.collection_name = collection_name
        self.dimension = dimension

        if path:
            self.client = qc.QdrantClient(path=path) 
        else:
            self.client = qc.QdrantClient(":memory:")

        match distance:
            case 'cosine': distancev = Distance.COSINE
            case 'l2': distancev = Distance.EUCLID
            case 'dot': distancev = Distance.DOT
            case _: raise ValueError(f'unknown metric {distance}')

        vparams = VectorParams(
                    size=self.dimension,
                    distance=distancev,
                )
        config = dict(collection_name=self.collection_name,
                    vectors_config = vparams,
                    #optimizers_config=qmodels.OptimizersConfigDiff(memmap_threshold=20000),
                    #hnsw_config=qmodels.HnswConfigDiff(on_disk=True)
                      )
        
        exists = self.client.collection_exists(collection_name)
        if (not exists) or recreate:
            self.client.recreate_collection(**config)

    
    def add(self, reps, paths):
        from qdrant_client.models import  Batch
        printd(2, f'add {len(paths)} to Qdrantdb')
        self.client.upsert(
            collection_name=self.collection_name,
            points=Batch(
                ids = [str(uuid.uuid5(uuid.NAMESPACE_OID, p)) for p in paths],
                vectors=reps,
                payloads=[dict(path=p) for p in paths]
            ),
        )

    def retrieve(self, rep_query, limit=None):
        if not isinstance(rep_query, list):
            rep_query = rep_query.tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=rep_query,
            #query_filter=_filter,
            limit=limit,
            with_payload=True,
        )

        docnodes = [
            ScoreNode(
                doc_path = res.payload['path'],
                score = res.score
            )
            for res in results
        ]

        return docnodes

class Storage:
    def __init__(self, config: StorageConfig) -> None:
        self.C = config
        print(f'init storage: {config}, {type(config)}')
        dbp = config.db_props
        from chromadb import PersistentClient

        match dbp.name:
            case 'chromadb':
                self.client = PersistentClient(path=dbp.path)
                self.collection = self.client.get_or_create_collection(self.C.collection_name)
            case 'qdrantdb':
                self.db = _QdrantDB(collection_name=self.C.collection_name, dimension=self.C.get_dimension(),
                                     path=dbp.path)
            case 'tensordb':
                self.collection = TensorCollection(dbpath=dbp.path, collection=self.C.collection_name)
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
        dbname = self.C.db_props.name
        printd(2, f'adding to storage {dbname} - paths: {paths[0]}-{paths[-1]}')
        match dbname:
            case 'chromadb':
                self.collection.add(
                    embeddings=[rep.tolist() for rep in reps], #expects a list of list
                    metadatas=[dict(path=p) for p in paths],
                    ids=paths
                )
            case 'tensordb':
                self.collection.add(reps, paths)
            
            case 'qdrantdb':
                self.db.add(reps, paths)
            case _:
                raise ValueError(f'Unsupported database {dbname}')

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
        dbname = self.C.db_props.name
        printd(2, f'retrieving from storage {dbname}')
        match dbname:
            case 'chromadb':
                docnodes = self.retrieve_chromadb(rep_query.tolist(), limit=limit)
            case 'tensordb':
                docnodes = self.retrieve_tensordb(rep_query, similarity_fn=similarity_fn, limit=limit)
            case 'qdrantdb':
                docnodes = self.db.retrieve(rep_query, limit=limit)
            case _:
                raise NotImplementedError(f'retrieve: {dbname}')
        return docnodes

