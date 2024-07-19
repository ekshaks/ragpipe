from ollama import Client
import os
from dotenv import load_dotenv

load_dotenv()  # Load the .env file

class LocalLLM:
  def __init__(self, host='http://192.168.0.171:11434') -> None:
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

local_llm = LocalLLM()


def groq_llm(prompt, model="groq/llama3-8b-8192"):
  from litellm import completion
  import os

  if 'GROQ_API_KEY' not in os.environ:
      raise("GROQ_API_KEY is not set in os.environ")

    
  response = completion(
        model=model, #"groq/llama3-70b-8192"
        messages=[
        {"role": "user", "content": prompt}
    ],
    #stream=True
    )
    #print(response)
  return response['choices'][0]['message']['content']


def respond_to_contextual_query(query, docs_retrieved, prompt, model='groq/mixtral-8x7b-32768'):
    docs_texts = '\n'.join([doc.get_text_content() for doc in docs_retrieved])
    prompt = prompt.format(documents=docs_texts, query=query)
    #resp = groq_llm(prompt, model='groq/llama3-70b-8192')
    resp = groq_llm(prompt, model=model)
    return resp