from typing import List, Optional


from .config import DBConfig, EncoderConfig
from .index import IndexConfig, IndexManager, ObjectIndex, RPIndex


from .common import  printd, load_func, detect_type
from .docnode import ScoreNode

from .db import StorageConfig

IM = IndexManager()
        

def create_or_load_index(docset: 'DocSet', repname: str, rep_props,
                        is_query=False, dbs=None):
    '''
    I = initialize_index_class(encoder_config, storage_config, docset, repname)
    - if ec.with_index: load ext library index
    - else: create RPIndex

    Key assumptions:
    - if 'store' is bool, then create a dynamic collection name based on docset
    - if 'store' is str, use the specified collection name


    '''
    from .encoders import get_encoder_reptype

    ec = encoder_config = rep_props.encoder
    store = rep_props.store
    collection_name = rep_props.collection
    rep_type = get_encoder_reptype(ec)
    
    if ec.with_index: 
        dbs = None
        index_type = 'custom_index'
    else:
        index_type = 'rpindex'

    if collection_name is None:
        collection_name = docset.collection_name_for(repname)

    if store not in [False, None]:
        assert not is_query, f'{rep_props}'
        storage_config = StorageConfig.from_kwargs(collection_name=collection_name, rep_type=rep_type, dbs=dbs, db_props=store) 
    else: 
        storage_config = None 
    printd(2, f'initialize_index: {docset.fpath}, {repname}, storage={storage_config}, encoder={encoder_config}')
    
    index_config = IndexConfig.from_kwargs(encoder_config, storage_config, docset.fpath, repname, docset.item_paths, index_type=index_type) 
    im_has_ic = IM.has(index_config) 
    index_exists = im_has_ic and store #load from existing index, only if store is true. unintuitive! FIX this
    if index_exists:
        printd(2, f'Found in IndexManager cache. {repname}, {encoder_config.name}')
    else: #build reps, create index, if storage_confadd key to IM
        printd(2, f'Not found in IndexManager cache: {repname}, {encoder_config.name}. Creating reps and index.')
    
    if ec.with_index:
        match ec.name:
            case 'bm25':
                from ext.libs.bm25 import RankBM25Index
                print('>>>> **building BM25 Index**')
                reps_index = RankBM25Index(docset.items, docset.item_paths, is_query=is_query) #TODO: Needs testing for query and docs cases
            case _:
                if ec.module is not None:
                    Indexer = load_func(ec.module)
                    reps_index = Indexer.from_index_config(index_config)
                    if index_exists:
                        reps_index.load()
                    else:
                        reps_index.add(docset, is_query=is_query)
                else:
                    raise ValueError(f'Unable to find enc-indexer {encoder_name}')
    else:
        reps_index = RPIndex.from_index_config(index_config)
        if not index_exists:
            #reps_index = encode_and_index(docset.items, index_config, is_query=is_query)
            reps_index.encode_and_index(docset, is_query=is_query)
    
    if store and not im_has_ic:
        IM.add(index_config)
    
    return reps_index



def compute_rep(fpath, D, dbs, rep_props=None, repname=None, is_query=False, doc_pre_filter=[]) -> 'Index':
    from .docstore import create_docset

    #fpath = .sections[].text repname = dense
    docset = create_docset(fpath, D, doc_pre_filter=doc_pre_filter)

    ## Encoder
    assert rep_props is not None
    #encoder_config = rep_props.encoder
    printd(2, f'compute_rep: props = {rep_props}, storage = {rep_props.store}')


    I = create_or_load_index(docset, repname, rep_props, is_query=is_query, dbs=dbs)

    return I

def retriever_router(doc_index, query_text, query_index, limit=10): #TODO: rep_query_path -> query_index
    '''
    Depending on the type of index, retrieve the doc nodes relevant to a query
    '''
    printd(2, f'retriever_router: doc_index type {type(doc_index)}, query_index type {type(query_index)}')
    #print(doc_index)
    #print(query_index)
    rep_query = query_index.get_query_rep()
    doc_nodes: List[ScoreNode] = doc_index.retrieve(rep_query, limit=limit) #only refs
    return doc_nodes