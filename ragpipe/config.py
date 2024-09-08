
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.fields import FieldInfo

from typing import Dict, List, Optional, Union, Literal, Any
from typing_extensions import Self
from pathlib import Path

'''
prompts: dict(str, str)
encoders: key -> EncoderConfig
representations: key -> RepConfig
bridges: key -> BridgeConfig
merges: key -> MergeConfig
'''

def split_str_to_list(v):
    if isinstance(v, str):
        # Split the string into a list if it's a string
        return [x.strip() for x in v.split(',')]
    return v

class EncoderShapeConfig(BaseModel, frozen=True):
    model_config = ConfigDict(extra='forbid')
    rep_type: str = 'single_vector'
    size: int = 384
    dtype: str = 'float32'
    
class EncoderConfig(BaseModel, frozen=True): 
    '''Combined config for Encoder and Indexer'''
    model_config = ConfigDict(extra='forbid')

    name: str
    prompt: Optional[str] = None
    query_instruction: Optional[str] = None

    with_index: bool = False
    module: Optional[str] = None #external module
    device: Optional[str] = 'cpu'
    batch_size: int = 1

    #TODO refactor below into EncoderShapeConfig
    dtype: Optional[str] = ''
    size: Optional[int] = None
    shape: Optional[EncoderShapeConfig] = None #Dict[str, EncoderShapeConfig] (allow multiple size, dtype)


class DBConfig(BaseModel, frozen=True):
    model_config = ConfigDict(extra='forbid')
    name: Literal['chromadb', 'qdrantdb', 'tensordb'] = 'chromadb' 
    path: str = '/tmp/ragpipe/'
    options: Optional[Dict[str, Dict]] = {}


class RepConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    encoder: Union[str, EncoderConfig]
    enabled: Optional[bool] = True
    store: Optional[Union[bool,str,DBConfig]] = False

    def update_encoder(self, encoders):
        #print('updating RepConfig for ', self.encoder, type(self.encoder))
        if isinstance(self.encoder, str):
            econfig = encoders.get(self.encoder, None)
            if econfig is None: 
                print('> enconfig None, call EncoderConfig')
                econfig = EncoderConfig(name=self.encoder)
            #raise ValueError(f'Unable to find encoder : {self.encoder}')
            self.encoder = econfig

    def update_store(self, store_map: Dict[str,DBConfig]):
        #if isinstance(self.store, bool): #handle after rep_type known
        if isinstance(self.store, str):
            try:
                self.store = store_map[self.store]
            except KeyError as e:
                print(f'No DB store defined: {self.store}')
                raise e


class BridgeConfig(BaseModel):
    repnodes: List[str] #sparse2, .paras[].text#sparse2
    limit: int
    enabled: Optional[bool] = True
    evalfn: Optional[str] = None 
    matchfn: Optional[str] = None

    @field_validator('repnodes', mode='before')
    @classmethod
    def split_names(cls, v: FieldInfo):
        return split_str_to_list(v)

class MergeConfig(BaseModel):
    limit: int
    bridges: Optional[List[str]] = []
    expr: Optional[str] = ''
    method: Optional[Literal['reciprocal_rank', 'expr']] = 'expr' 

    @field_validator('bridges', mode='before')
    @classmethod
    def split_names(cls, v: FieldInfo):
        return split_str_to_list(v)
    
    def update_bridges_from_expr(self):
        if self.method == 'expr':
            import sympy as sp
            e = sp.sympify(self.expr)
            bs = list(map(str, e.free_symbols))
            self.bridges.extend(bs)


class RPConfig(BaseModel):
    config_fname: str 
    prompts: Optional[Dict[str, str]] = {} #name to prompt
    encoders: Optional[Dict[str, EncoderConfig]] = {}
    dbs: Optional[Dict[str, DBConfig]] = {}
    llm_models: Optional[Dict[str, str]] = {}
    representations: Dict[str,  Dict[str, RepConfig] ] #doc field -> repname -> repconfig 
    bridges: Dict[str, BridgeConfig]
    merges: Optional[Dict[str, MergeConfig]] = {}
    enabled_merges: Optional[List] = Field(default_factory=list)
    queries: Optional[List[str]] = []
    etc: Optional[Dict[str,Any]] = {}

    @field_validator('enabled_merges', mode='before')
    @classmethod
    def split_names(cls, v: FieldInfo):
        return split_str_to_list(v)
    

    @model_validator(mode='after')
    def model_normalize(self) -> Self:
        dbs = self.dbs
        dbs['__default_single_vector__'] = DBConfig()
        dbs['__default_multi_vector__'] = DBConfig(name='tensordb')

        llm_models = self.llm_models
        if len(llm_models) == 0:
            llm_models['__default__'] = 'groq/llama3-70b-8192'

        em = self.enabled_merges
        if len(em) == 0: #no merge selected
            merges = self.merges
            num_merges = len(merges)
            if num_merges == 0: #create a default merge
                first_bridge = list(self.bridges.keys())[0]
                merge = MergeConfig(method='expr', expr=first_bridge, limit=20)
                m1 = '_rp_m1_'
                self.merges[m1] = merge
                self.enabled_merges = [m1]
            else:
                keys = list(merges.keys())
                if num_merges == 1: 
                    self.enabled_merges = keys
                else:
                    raise ValueError(f'Please select at least one of the merges (use "enabled_merges"): {keys}.')
        
        #add inbuilt encoders to self.encoders
        if 'bm25' not in self.encoders:
            BM25config = EncoderConfig(name='bm25', with_index=True)
            self.encoders['bm25'] = BM25config
        # replace enc name with enc config
        for field_rep in self.representations.values():
            for repname, repconfig in field_rep.items():
                repconfig.update_encoder(self.encoders)
                repconfig.update_store(dbs)
        
        for mk, mc in self.merges.items():
            mc.update_bridges_from_expr()
        return self


import typer
appt = typer.Typer()

def deep_update(source, overrides):
    """Recursively update the source dictionary with overrides."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source:
            source[key] = deep_update(source.get(key, {}), value)
        else:
            source[key] = value
    return source

def load_config(source, is_file=True, overrides='overrides.yaml', show=False):
    import yaml
    if is_file:
        with open(source, 'r') as file:
            configd = yaml.load(file, Loader=yaml.FullLoader)
            configd['config_fname'] = source
    elif isinstance(source, str):
        from .common import generate_uuid_from_string
        configd = yaml.safe_load(source)
        configd['config_fname'] = generate_uuid_from_string(source)

    else:
        raise ValueError(f'Unknown config source: {source}')


    opath = Path(source).parent / overrides
    if opath.exists():
        with open(opath, 'r') as file:
            overrides_configd = yaml.load(file, Loader=yaml.FullLoader)
        configd = deep_update(configd, overrides_configd)
    
    config = RPConfig(**configd)
    
    if show:
        from pprint import pprint
        pprint(config.model_dump(exclude_none=True))
    return config

@appt.command()
def load_config_cmd(fname, show: bool = typer.Option(False, "--show")):
    return load_config(fname, show=show)

if __name__ == '__main__':
    appt()