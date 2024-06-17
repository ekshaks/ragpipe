import re
import json
import yaml
from pathlib import Path

from .common import DotDict, printd
from .llms import local_llm

# Globals

prompts_file_path = Path(__file__).resolve().parent / 'prompts.yaml' 
with open(prompts_file_path, 'r') as file:
    PROMPTS = DotDict(yaml.safe_load(file))

def query_decomposer(query, prompt=None):
    if prompt is None:
        prompt = PROMPTS.query_decomposer.format(query=query)
    resp = local_llm.__call__(prompt)
    print(resp)
    resp = DotDict(json.loads(resp))
    return resp

def transform(text_list, encoder_name, prompt=None, is_query=True):
    match encoder_name:
        case 'llm/query_decomposer':
            printd(3, f'encoding with llm/query_decomposer: {text_list}')
            return [query_decomposer(text, prompt=prompt) for text in text_list]
        case _:
            raise ValueError(f'unknown {encoder_name}')


if __name__ == '__main__':
    #read_data()
   query_decomposer("What's the net worth of the fourth richest billionaire in 2023?")