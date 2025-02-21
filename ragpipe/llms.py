import os
from dotenv import load_dotenv
from .common import printd
from pydantic import BaseModel
from pathlib import Path
from typing import Dict, List, Any, Union

load_dotenv()  # Load the .env file

class LLMOp(BaseModel):
  model: str
  prompt: str #instructions, with no placeholders
  #image_paths: List[Union[str, Path]] #data separate
  params: Dict[str, Any]

class LocalLLM:
  def __init__(self, host='http://localhost:11434') -> None:
    from ollama import Client
    self.client = Client(host=host)
  
  def __call__(self, prompt, model=None):
    model = 'mistral:instruct' if model is None else model
    response = self.client.chat(model=model, messages=[
    {
      'role': 'user',
      'content': prompt,
    }
    ])
    return response['message']['content']

def validate_model_keys(model):
  import os
  prefix_key_pairs = [
     ('groq/', 'GROQ_API_KEY'),
     ('openai/', 'OPENAI_API_KEY'),
     ('gemini/', 'GEMINI_API_KEY')

  ]

  for prefix, api_key in prefix_key_pairs:
    if model.startswith(prefix) and api_key not in os.environ:
        raise ValueError(f"{api_key} is not set in os.environ, or ragpipe/.env")

def cloud_llm(prompt, model=None):
  from litellm import completion
  model = model or "groq/llama3-8b-8192"
  validate_model_keys(model)
  response = completion(
        model=model, #"groq/llama3-70b-8192"
        messages=[
        {"role": "user", "content": prompt}
    ],
    #stream=True
    )
    #print(response)
  return response['choices'][0]['message']['content']


def encode_image(image_path):
  import base64
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def vlm_template_apply(prompt, image_paths, **kwargs):
  txt_content = [dict(type="text", text=prompt)]
  max_images_per_call = kwargs.get('max_images_per_call')
  img_content = [dict(type="image_url", 
                      image_url=dict(url=f"data:image/jpeg;base64,{encode_image(image_path)}") 
                      ) 
                 for image_path in image_paths[:max_images_per_call]]
  content = txt_content + img_content
  messages = [dict(role='user', content=content)]
  return messages
  
def _cloud_vlm(image_paths, prompt, model=None, **kwargs):
  from litellm import completion
  model = model or "groq/llama-3.2-11b-vision-preview"
  validate_model_keys(model)
  messages = vlm_template_apply(prompt, image_paths, **kwargs)
  params = dict(
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=False,
    stop=None,
    #response_format={"type": "json_object"},
  )
  response = completion(model=model, messages=messages, **params)
  return response['choices'][0]['message']['content']

def cloud_vlm(image_paths, op: LLMOp):
  return _cloud_vlm(image_paths, op.prompt, op.model, **op.params)



### higher level API


def llm_router(prompt, model=None, config=None):
   if model and\
      (model.startswith('local/') or model.startswith('ollama/')):
          
      ollama_host = None 
      if config is not None:
        ollama_host = config.llm_models.get('ollama_host')
      
      local_llm = LocalLLM(host=ollama_host)
      parts = model.split('/', 1)
      if len(parts) < 1: raise ValueError(f'Invalid model name: {model}')
      return local_llm(prompt, model=parts[1])
   else:
      return cloud_llm(prompt, model=model)
      
      
def respond_to_contextual_query(query, docs_retrieved, prompt_templ, model=None, config=None):
    from .prompts import eval_template
    docs_texts = '\n'.join([doc.get_text_content() for doc in docs_retrieved])
    prompt = eval_template(prompt_templ, documents=docs_texts, query=query)
    if model is None:
      if config is not None:
        model = config.llm_models['__default__']
      else:
        raise ValueError('''respond_to_contextual_query: No model specified. Either specify the `model` parameter or pass in the `config` parameter.
                         ''')
    resp = llm_router(prompt, model=model, config=config)
    return resp


def test():
    #from ragpipe.llms import cloud_llm, llm_router
    resp = llm_router('are you there?')
    print(resp)
    resp = llm_router('are you there?', model='ollama/llama3.1')
    print(resp)

if __name__ == '__main__':
   test()