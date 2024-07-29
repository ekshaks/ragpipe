from typing import List

#from llama_index.core.schema import TextNode
#from llama_index.core import load_index_from_storage

from .index import IndexConfig, IndexManager, ObjectIndex, RPIndex, VectorStoreIndexPath


from .common import get_fpath_items, fpath2collection, printd
from .docnode import ScoreNode

from .db import StorageConfig

IM = IndexManager()

from . import llm_bridge
from .encoders import BM25, get_encoder

def encode_and_index(encoder, repname, 
                 items, item_paths, 
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

            reps_index = RPIndex(encoder=encoder, storage_config=storage_config)
            reps_index.add(docs=items, doc_paths=item_paths, is_query=is_query)

        case 'objindex':
            reps_index = ObjectIndex(reps=reps, paths=item_paths, is_query=is_query, 
                                        docs_already_encoded=True) #

        case 'noindex':
            pass
        case _:
            raise ValueError(f"unknown index: {index_type}")
        
    return reps_index


def compute_rep(fpath, D, rep_props=None, repname=None, is_query=False) -> '*Index':
    #fpath = .sections[].text repname = dense
    assert rep_props is not None
    encoder_config = rep_props.encoder
    storage = rep_props.store
    printd(3, f'compute_rep: props = {rep_props}, storage = {storage}')
  
    ##encoder model loader, index_type, rep_type
    doc_leaf_type = D.get('doc_leaf_type', 'raw')
    #encoder_config = dict(doc_leaf_type=doc_leaf_type) #TODO: from props?
    encoder = get_encoder(encoder_config, doc_leaf_type=doc_leaf_type) #

    storage_config = None if not storage else StorageConfig(collection_name=fpath2collection(fpath,repname), 
                                                            rep_type=encoder.rep_type)
    print(fpath, repname, f': storage={storage_config}, encoder={encoder_config}')

    index_config = IM.get_config(fpath, repname, encoder_config) if storage_config else None
    
    if index_config is None:
        printd(2, f'Not found in IndexManager cache: {repname}, {encoder_config.name}. Creating reps.')

        items_path_pairs = get_fpath_items(fpath, D)
        items, item_paths = items_path_pairs.els, items_path_pairs.paths

        index_type = 'llamaindex' if doc_leaf_type == 'llamaindex' else 'rpindex'
        reps_index = encode_and_index(encoder, repname, 
                    items, item_paths,    
                    storage_config, is_query=is_query, index_type=index_type)
        if storage_config is not None: #does making a query rpindex make sense? change query?
            IM.add(fpath, repname, encoder_config, reps_index)
        else:
            printd(2, f'storage_config is None - not creating index.')

    else:

        printd(2, f'Found in IndexManager cache: {repname}, {encoder_config.name}.')
        printd(3, f'{index_config}')
        match index_config.index_type:
            case 'llamaindex':
                reps_index = VectorStoreIndexPath.from_index_config(index_config)
            case 'rpindex':
                reps_index = RPIndex.from_index_config(index_config)
            case _ :
                raise ValueError(f'unknown index type {index_config.index_type}')
    #printd(2, f'compute_rep: {fpath} -> {reps_index}')
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
        case VectorStoreIndexPath():
            query_bundle = QueryBundle(query_str=query_text, embedding=rep_query)
            retriever = VectorIndexRetriever(index=doc_index.get_vector_store_index(), similarity_top_k=limit)
            li_nodes = retriever.retrieve(query_bundle)
            #vector_ids = {n.node.node_id for n in vector_nodes}
            doc_nodes = [ScoreNode(li_node=n, score=n.score) for n in li_nodes]

            return doc_nodes

        case RPIndex():
            doc_nodes: List[ScoreNode] = doc_index.retrieve(rep_query, limit=limit) #only refs
            return doc_nodes
        
        case _:
            raise NotImplementedError(f'unknown index: {doc_index}')
