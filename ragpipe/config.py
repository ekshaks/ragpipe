
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.fields import FieldInfo

from typing import Dict, List, Optional, Union, Literal
from typing_extensions import Self

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

class EncoderConfig(BaseModel, frozen=True): 
    model_config = ConfigDict(extra='forbid')

    name: str
    dtype: Optional[str] = ''
    size: Optional[int] = None
    prompt: Optional[str] = None
        

class RepConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

    encoder: Union[str, EncoderConfig]
    enabled: Optional[bool] = True
    store: Optional[bool] = False

    def update_encoder(self, encoders):
        if isinstance(self.encoder, str):
            econfig = encoders.get(self.encoder, None)
            if econfig is None: 
                econfig = EncoderConfig(name=self.encoder)
            #raise ValueError(f'Unable to find encoder : {self.encoder}')
            self.encoder = econfig

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

class RPConfig(BaseModel):
    prompts: Optional[Dict[str, str]] = {} #name to prompt
    encoders: Optional[Dict[str, EncoderConfig]] = {}
    llm_models: Optional[Dict[str, str]] = {}
    representations: Dict[str,  Dict[str, RepConfig] ] #doc field -> repname -> repconfig 
    bridges: Dict[str, BridgeConfig]
    merges: Dict[str, MergeConfig]
    enabled_merges: Optional[List] = Field(default_factory=list)
    queries: Optional[List[str]] = []

    @field_validator('enabled_merges', mode='before')
    @classmethod
    def split_names(cls, v: FieldInfo):
        return split_str_to_list(v)
    

    @model_validator(mode='after')
    def model_normalize(self) -> Self:

        em = self.enabled_merges
        #print('at least one merge: ', len(em))
        if len(em) == 0: #empty list
            merges = self.merges
            keys = list(merges.keys())
            if len(merges) == 0: 
                raise ValueError(f'Please specify at least one merge.')
            if len(merges) == 1:
                self.enabled_merges = keys
            else:
                raise ValueError(f'Please select at least one of the merges (use "enabled_merges"): {keys}.')
        
        # replace enc name with enc config
        for field_rep in self.representations.values():
            for repname, repconfig in field_rep.items():
                repconfig.update_encoder(self.encoders)
        
        return self


import typer
appt = typer.Typer()

def load_config(fname, show=False):
    import yaml
    with open(fname, 'r') as file:
        config = RPConfig(**yaml.load(file, Loader=yaml.FullLoader))
    
    if show:
        from pprint import pprint
        pprint(config.model_dump(exclude_none=True))
    return config

@appt.command()
def load_config_cmd(fname, show: bool = typer.Option(False, "--show")):
    return load_config(fname, show=show)

if __name__ == '__main__':
    appt()