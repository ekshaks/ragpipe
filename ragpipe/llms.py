from ollama import Client

class LLM:
  def __init__(self, host='http://192.168.0.171:11434') -> None:
    self.client = Client(host=host)
  
  def call_api(self, prompt, model=None):
    model = 'mistral:instruct' if model is None else model
    response = self.client.chat(model=model, messages=[
    {
      'role': 'user',
      'content': prompt,
    }
    ])
    return response['message']['content']
  