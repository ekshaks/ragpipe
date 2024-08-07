from typing import List

#from llama_index.core.schema import TextNode
#from llama_index.core import load_index_from_storage

from .index import IndexConfig, IndexManager, ObjectIndex, RPIndex


from .common import get_fpath_items, get_collection_name, printd
from .docnode import ScoreNode

from .db import StorageConfig

IM = IndexManager()

from . import llm_bridge
from .encoders import BM25, get_encoder

def encode_and_index(encoder, fpath, repname, items, item_paths, 
                 storage_config, is_query=False, index_type='rpindex'):
    
    encoder_name = encoder.name
    
    if encoder_name.startswith('llm'):
        prompt = encoder.config.prompt
        reps = llm_bridge.transform(items, encoder_name, prompt=prompt, is_query=is_query)
        index_type = 'objindex'

    elif encoder_name == 'passthrough':
        print('computing rep passthrough')
        _repname = repname[1:] # _header -> header
        reps = [item[_repname] for item in items]
        index_type = 'objindex'
    else: pass

    match index_type:
        case 'llamaindex':
            assert isinstance(items, list) and len(items) > 0, f"items is {type(items)}"
            ic = IndexConfig(index_type=index_type, encoder_config=encoder.config, 
                doc_paths=item_paths, storage_config=storage_config)
            reps_index = VectorStoreIndexPath.from_docs(items, ic, encoder_model=encoder.get_model())
            
        case 'rpindex':
            item_type = type(items[0]).__name__
            if item_type != 'str': #handle LI text nodes. TODO: what if LI documents?
                assert 'TextNode' in item_type, f'Expected item {repname} of type str, but found {item_type}'
                items = [item.text for item in items]

            ic = IndexConfig(encoder_config=encoder.config, 
                             storage_config=storage_config,
                             fpath=fpath, repname=repname,
                             doc_paths=item_paths)
            reps_index = RPIndex(ic)
            reps_index.add(docs=items, doc_paths=item_paths, is_query=is_query)

        case 'objindex':
            reps_index = ObjectIndex(reps=reps, paths=item_paths, is_query=is_query, 
                                        docs_already_encoded=True) #

        case 'noindex':
            pass
        case _:
            raise ValueError(f"unknown index: {index_type}")
        
    return reps_index


def compute_rep(fpath, D, dbs, rep_props=None, repname=None, is_query=False) -> '*Index':
    #fpath = .sections[].text repname = dense
    assert rep_props is not None
    encoder_config = rep_props.encoder
    printd(3, f'compute_rep: props = {rep_props}, storage = {rep_props.store}')
  
    ##encoder model loader, index_type, rep_type
    doc_leaf_type = D.get('doc_leaf_type', 'raw')
    #encoder_config = dict(doc_leaf_type=doc_leaf_type) #TODO: from props?
    encoder = get_encoder(encoder_config, doc_leaf_type=doc_leaf_type) #

    storage_config = None if not rep_props.store else\
          StorageConfig.from_kwargs(
            collection_name=get_collection_name(fpath,repname), 
            rep_type = encoder.rep_type,
            dbs=dbs, #lookup its db from list
            db_props=rep_props.store
         )
    
    print(fpath, repname, f': storage={storage_config}, encoder={encoder_config}')
    items_path_pairs = get_fpath_items(fpath, D)
    items, item_paths = items_path_pairs.els, items_path_pairs.paths

    index_config = IndexConfig.from_kwargs(encoder_config, storage_config, fpath, repname, item_paths) 
    #check if index key exists in IM, index created already
    #note index_config depends on item_paths but storage_config doesn't. 
    # so can't check storage_collection name exists only
    index_exists = IM.has(index_config)
    if index_exists: #key exists, load index
        reps_index = RPIndex.from_index_config(index_config)
        printd(2, f'Found in IndexManager cache: {index_config}.')
    else: #build reps, create index, if storage_confadd key to IM
        printd(2, f'Not found in IndexManager cache: {repname}, {encoder_config.name}. Creating reps.')
        #TODO: replace as many args by index_config
        reps_index: '..Index' = encode_and_index(encoder, fpath, repname, items, item_paths,
                    storage_config, is_query=is_query, index_type='rpindex')
        
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
    assert(isinstance(query_index, RPIndex)), f'query_index type= {type(query_index)} '
    rep_query = query_index.get_query_rep()

    match doc_index:
        # case VectorStoreIndexPath():
        #     query_bundle = QueryBundle(query_str=query_text, embedding=rep_query)
        #     retriever = VectorIndexRetriever(index=doc_index.get_vector_store_index(), similarity_top_k=limit)
        #     li_nodes = retriever.retrieve(query_bundle)
        #     #vector_ids = {n.node.node_id for n in vector_nodes}
        #     doc_nodes = [ScoreNode(li_node=n, score=n.score) for n in li_nodes]

        #     return doc_nodes

        case RPIndex():
            doc_nodes: List[ScoreNode] = doc_index.retrieve(rep_query, limit=limit) #only refs
            return doc_nodes
        
        case _:
            raise NotImplementedError(f'unknown index: {doc_index}')
