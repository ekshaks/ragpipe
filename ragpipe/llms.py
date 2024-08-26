import os
from dotenv import load_dotenv

load_dotenv()  # Load the .env file

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



def cloud_llm(prompt, model=None):
  from litellm import completion
  import os

  model = model or "groq/llama3-8b-8192"

  prefix_key_pairs = [
     ('groq/', 'GROQ_API_KEY'),
     ('openai/', 'OPENAI_API_KEY')
  ]

  for prefix, api_key in prefix_key_pairs:
    if model.startswith(prefix) and api_key not in os.environ:
        raise ValueError(f"{api_key} is not set in os.environ, or ragpipe/.env")

    
  response = completion(
        model=model, #"groq/llama3-70b-8192"
        messages=[
        {"role": "user", "content": prompt}
    ],
    #stream=True
    )
    #print(response)
  return response['choices'][0]['message']['content']

def llm_router(prompt, model=None, ollama_host=None):
   if model and\
      (model.startswith('local/') or model.startswith('ollama/')):
      
      local_llm = LocalLLM(host=ollama_host)
      parts = model.split('/', 1)
      if len(parts) < 1: raise ValueError(f'Invalid model name: {model}')
      return local_llm(prompt, model=parts[1])
   else:
      return cloud_llm(prompt, model=model)
      
      
def respond_to_contextual_query(query, docs_retrieved, prompt_templ, model='groq/mixtral-8x7b-32768'):
    from .prompts import eval_template
    docs_texts = '\n'.join([doc.get_text_content() for doc in docs_retrieved])
    prompt = eval_template(prompt_templ, documents=docs_texts, query=query)
    resp = llm_router(prompt, model=model)
    return resp


def test():
    #from ragpipe.llms import cloud_llm, llm_router
    resp = llm_router('are you there?')
    print(resp)
    resp = llm_router('are you there?', model='ollama/llama3.1')
    print(resp)

if __name__ == '__main__':
   test()