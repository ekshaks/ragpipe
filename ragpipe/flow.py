from pydantic import BaseModel
from typing import Optional, List
from .config import BridgeConfig, RepConfig, RPConfig
from .common import printd, load_func, DEFAULT_LIMIT, DotDict, has_field
from .docnode import ScoreNode


class RepManager:
    '''RepConfig
    encoder: Union[str, EncoderConfig]
    enabled: Optional[bool] = True
    store: Optional[Union[bool,str,DBConfig]] = False
    
    representations: Dict[str,  Dict[str, RepConfig] ] #doc field -> repname -> repconfig 
    
    '''

    def __init__(self, config: RPConfig):
        self.config = config
        self.dbs = config.dbs
        self.reps = {}
    
    def hash_field_repname(self, fpath, repname):
        return f'{fpath}#{repname}'
    def decomp_field_repname(self, repkey):
        return repkey.split('#')
    
    def create_rep(self, D, fpath, repname, config: RepConfig, doc_pre_filter=[]):
        from .rag_components import compute_rep
        is_query = fpath.startswith('qu') #hack! need a flag
        rep = compute_rep(fpath, D, self.dbs, 
                            rep_props=config, repname=repname, is_query=is_query, doc_pre_filter=doc_pre_filter)
        return rep
    
    def clear_rep(self, repkey):
        self.reps.pop(repkey, None)
    
    def clear_all_reps_fpath(self, fpath):
        clear_keys = []
        for k in self.reps.keys():
            if fpath in k:
                clear_keys.append(k)
        for k in clear_keys: self.clear_rep(k)

    def get_or_create_rep(self, repkey, D, doc_pre_filter=[]):
        printd(1, f'\n ~~~~ computing reps for {repkey}...')
        CReps = self.config.representations
        #TODO: turn repkey to doc_paths. hash(doc_paths \int doc_pre_filter), use as hash as key in self.reps
        if repkey not in self.reps:
            fpath, repname = self.decomp_field_repname(repkey)
            try:
                repconfig = CReps[fpath][repname]
                rep = self.create_rep(D, fpath, repname, repconfig, doc_pre_filter=doc_pre_filter)
                self.reps[repkey] = rep
            except Exception as e:
                print(f'Unable to resolve repkey {repkey}. Did you define rep config for {repkey} correctly?')
                if CReps.get(fpath) is None:
                    print(f'No rep config found for {fpath}.')
                else:
                    print(f'Defined rep names for {fpath} are: {list(CReps[fpath].keys())}')
                raise e

            # rep_value = list(self.representations.values())[0]
            # try:
            #     #printd(1, f'~~~ computing {repkey}')
            #     if isinstance(rep_value, dict): #older style: fpath -> repname -> repconfig (TODO: remove)
            #         repconfig = rep_value[repname]
            #     else: #repname -> repconfig
            #         repconfig = rep_value
            #     rep = self.create_rep(D, fpath, repname, repconfig, doc_pre_filter=doc_pre_filter)
            #     self.reps[repkey] = rep

        else:
            printd(1, f' !! ---> rep {repkey} already computed. Reusing it.') 
        
        return self.reps[repkey]


def create_rep_manager(config: RPConfig): 
    '''
    Only create right now. TODO: cache strategy?
    RMPool: Dict[str, RepManager] = {}
    config_fname = config.config_fname
    RM = RMPool.get(config_fname, None)
    if RM is None:
    '''
    RM = RepManager(config)
    #    RMPool[config_fname] = RM
    return RM


class BridgeRetriever:

    '''
    repnodes: List[str] #sparse2, .paras[].text#sparse2
    limit: int
    enabled: Optional[bool] = True
    evalfn: Optional[str] = None 
    matchfn: Optional[str] = None

    b_sparse:
      repnodes: query.text#sparse, .paras[].text#sparse
      limit: 15
      enabled: false
    '''

    def __init__(self, bridge_name, config: RPConfig, RM: RepManager = None):
        self.name = bridge_name
        self._config = config
        self.bconfig = config.bridges[bridge_name]
        self.RM = RM or create_rep_manager(config)
        # clear query reps. TODO: here or at callee?
        # self.RM.clear_all_reps_fpath('query')

    def eval(self, query_text, D, doc_pre_filter: List[ScoreNode]=[], **kwargs):
        printd(2, f'\n===== Start eval Bridge({self.name}) =====\nbridge config = {self.bconfig}\n')
        if not has_field(D, 'query'):
            D.query = DotDict(text=query_text)

        repnodes = self.bconfig.repnodes
        assert isinstance(repnodes, list) and len(repnodes) == 2, f'{repnodes}'
        limit = self.bconfig.limit or DEFAULT_LIMIT

        # create reps
        reps = [self.RM.get_or_create_rep(repkey, D, doc_pre_filter=doc_pre_filter) 
                for repkey in repnodes] #generally only 2 elements in repnodes (query..., doc...)

        printd(2, f'\n ~~~~ Retrieving docs for Bridge {self.name}...')
        matchfn_key = self.bconfig.matchfn
        if matchfn_key is not None:
            matchfn = load_func(matchfn_key)
            docs: List[ScoreNode] = matchfn(*reps)
        else:
            # match and retrieve from indices
            from .rag_components import retriever_router
            query_index, doc_index = reps
            docs: List[ScoreNode] = retriever_router(doc_index, D.query.text, query_index, limit=limit)
        
        evalfn_key = self.bconfig.evalfn
        if evalfn_key is not None:
            evalfn = load_func(evalfn_key)
            evalfn(docs, D, query_id=kwargs.get('query_id', None))
        printd(2, f'\n===== End eval Bridge({self.name}) =====\n')
        return docs

       
class Retriever:

    def __init__(self, rp_config: RPConfig, RM: RepManager = None):
        self.config = rp_config
        self.RM = RM or create_rep_manager(rp_config)
    
    def merge_by_expr(self, expr, bridge2docs, merge=None):
        #move to fusion.py?
        import sympy as sp
        e = sp.sympify(expr)
        bs = list(map(str, e.free_symbols))
        #use expr to gen new scores for each doc common across all bridge_names. sort.
        if len(bs) == 1:
            return bridge2docs[bs[0]]
        else:
            raise NotImplementedError(f'TODO: merge multiple result lists using expressions. bridges = {bs}') 
    
    def eval(self, query_text, D, 
             merge:str = None, query_id:int = None, 
             doc_pre_filter: List[ScoreNode]=[]):
        #printd(1, f'\n=== Fusing Results, Ranking ... merges = {selected_merges}\n')

        D.query = DotDict(text=query_text)
        C: RPConfig = self.config

        if merge is None:
            if C.enabled_merges:
                merge = C.enabled_merges[0]
            else:
                values = list(C.merges.values())
                if values: merge = values[0] 
                else: raise ValueError(f'No merge specified.')

        mp = C.merges[merge]  
        bridge2results = {}
        for b in mp.bridges:
            br = BridgeRetriever(b, C, RM=self.RM)
            docs = br.eval(query_text, D, query_id=query_id, doc_pre_filter=doc_pre_filter)
            bridge2results[b] = docs

        print(f'\n==== Computing merge {merge} ==== ')   

        match mp.method:
            case 'reciprocal_rank':
                from .fusion import reciprocal_rank_fusion
                doc_with_scores = reciprocal_rank_fusion(bridge2results)[:mp.limit]
            case 'expr':
                doc_with_scores = self.merge_by_expr(mp.expr, bridge2results)[:mp.limit]
            case _:
                raise NotImplementedError(f'Unknown merge method : {mp.method}\nmerge_config: {mp}')
        
        doc_with_scores = [d for d in doc_with_scores if d.load_docs(D)]

        print(f'\n==== Done computing merge {merge} ==== \n')   

        
        return doc_with_scores

