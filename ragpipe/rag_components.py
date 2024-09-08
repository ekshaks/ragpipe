from typing import List

#from llama_index.core.schema import TextNode
#from llama_index.core import load_index_from_storage

from .index import IndexConfig, IndexManager, ObjectIndex, RPIndex


from .common import get_fpath_items, get_collection_name, printd, load_func, detect_type
from .docnode import ScoreNode

from .db import StorageConfig

IM = IndexManager()


#def encode_and_index(encoder, fpath, repname, items, item_paths, 
#                 storage_config, is_query=False, index_type='rpindex'):
def encode_and_index(items, ic: IndexConfig, is_query=False):
    ec = ic.encoder_config
    encoder_name = ec.name
    item_paths = ic.doc_paths

    if not is_query and ec.with_index:
        print('encode_and-index: econfig = ', ec)
        match encoder_name:
            case 'bm25':
                from ext.libs.bm25 import RankBM25Index
                print('>>>> **building BM25 Index**')
                reps_index = RankBM25Index(items, item_paths)
            case _:
                if ec.module is not None:
                    Indexer = load_func(ec.module)
                    reps_index = Indexer()
                    reps_index.add(items, item_paths, is_query=is_query)
                else:
                    raise ValueError(f'Unable to find enc-indexer {encoder_name}')

    else:
        if not isinstance(items[0], str): #handle LI text nodes. TODO: what if LI documents?
            item_type = type(items[0]).__name__
            
            if 'TextNode' in item_type: #legacy, TODO: remove!
                items = [item.text for item in items]
            else:
                item_type = detect_type(items[0])
                assert item_type != 'Unknown'


        reps_index = RPIndex(index_config=ic)
        reps_index.add(docs=items, doc_paths=item_paths, is_query=is_query)
        
    return reps_index


def compute_rep(fpath, D, dbs, rep_props=None, repname=None, is_query=False, doc_pre_filter=[]) -> 'Index':
    from .encoders import get_encoder_reptype

    #fpath = .sections[].text repname = dense
    assert rep_props is not None
    encoder_config = rep_props.encoder
    printd(3, f'compute_rep: props = {rep_props}, storage = {rep_props.store}')
  
    ##encoder model loader, index_type, rep_type
    doc_leaf_type = D.get('doc_leaf_type', 'raw')
    rep_type = get_encoder_reptype(encoder_config)

    storage_config = None if not rep_props.store else\
          StorageConfig.from_kwargs(
            collection_name=get_collection_name(fpath,repname), 
            rep_type = rep_type,
            dbs=dbs, #lookup its db from list
            db_props=rep_props.store
         )
    
    print(fpath, repname, f': storage={storage_config}, encoder={encoder_config}')
    dpfilter = set([d.doc_path for d in doc_pre_filter])
    items_path_pairs = get_fpath_items(fpath, D, docpath_pre_filter=dpfilter)
    items, item_paths = items_path_pairs.els, items_path_pairs.paths

    index_config = IndexConfig.from_kwargs(encoder_config, storage_config, fpath, repname, item_paths) 
    #check if index key exists in IM, index created already
    #note index_config depends on item_paths but storage_config doesn't. 
    # so can't check storage_collection name exists only
    index_exists = IM.has(index_config)
    if index_exists: #key exists, load index
        reps_index = RPIndex.from_index_config(index_config)
        printd(2, f'Found in IndexManager cache. {repname}, {encoder_config.name}')
    else: #build reps, create index, if storage_confadd key to IM
        printd(2, f'Not found in IndexManager cache: {repname}, {encoder_config.name}. Creating reps.')
        #TODO: replace as many args by index_config
        #reps_index: '..Index' = encode_and_index(encoder, fpath, repname, items, item_paths,
        #            storage_config, is_query=is_query, index_type='rpindex')
        reps_index = encode_and_index(items, index_config, is_query=is_query)
        
        if storage_config is not None: #does making a query rpindex make sense? change query?
            printd(2, f'... adding to index.')
            IM.add(index_config)
        else:
            printd(2, f'... not adding to index.')


    return reps_index

def retriever_router(doc_index, query_text, query_index, limit=10): #TODO: rep_query_path -> query_index
    '''
    Depending on the type of index, retrieve the doc nodes relevant to a query
    '''
    from llama_index.core import QueryBundle
    from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
    printd(3, f'retriever_router: doc_index type {type(doc_index)}, {type(query_index)}')
    #print(doc_index)
    #print(query_index)
    rep_query = query_index.get_query_rep()
    doc_nodes: List[ScoreNode] = doc_index.retrieve(rep_query, limit=limit) #only refs
    return doc_nodes