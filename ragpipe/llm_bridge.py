import re
import json
import yaml
from pathlib import Path

from .common import DotDict, printd
from .llms import llm_router
from .prompts import eval_template

# Globals

prompts_file_path = Path(__file__).resolve().parent / 'prompts.yaml' 
with open(prompts_file_path, 'r') as file:
    PROMPTS = DotDict(yaml.safe_load(file))

def query_decomposer(query, model, prompt_templ=None):
    if prompt_templ is None:
        prompt_templ = PROMPTS.query_decomposer

    prompt = eval_template(prompt_templ, query=query)
    
    resp = llm_router(prompt, model=model)
    print(resp)
    resp = DotDict(json.loads(resp))
    return resp

def transform(text_list, encoder_name, model, prompt=None, is_query=True):
    match encoder_name:
        case 'llm/query_decomposer':
            printd(3, f'encoding with llm/query_decomposer: {text_list}')
            return [query_decomposer(text, model, prompt_templ=prompt) for text in text_list]
        case _:
            raise ValueError(f'unknown {encoder_name}')


if __name__ == '__main__':
    #read_data()
   query_decomposer("What's the net worth of the fourth richest billionaire in 2023?")