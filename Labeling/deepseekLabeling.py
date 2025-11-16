import requests
import json
import os
from utils import LabelingClass
get_content = lambda x: x['choices'][0]['message']['content']

class DeepSeekLabeling(LabelingClass):

  def __init__(self):
    super().__init__()
    self.request_template = lambda prompt:{
      'url': "https://openrouter.ai/api/v1/chat/completions",
      'headers': {"Authorization": f"Bearer {os.getenv("OPEN_ROUTER_API_KEY")}"},
      'data': json.dumps({
        "model": "meta-llama/llama-4-maverick:free",
          "messages": [
            {
              "role": "system",
              "content": self.instruction
            },
            {
              "role": "user",
              "content": prompt
            }
          ],
      })
    }

  def _send_request(self, prompt):
    if prompt == None:
      raise ValueError("Prompt cannot be None")
    attempt = 0
    while attempt < self.MAX_ATTEMPTS:
      response = requests.post(**self.request_template(prompt)
      )
      if response.status_code != 200:
        raise ValueError(f"Request failed with status code {response.status_code}: {response.text}")
      
      json.dump(response.json(), open("tmp/deepseek_debug.json","w"), indent=2, ensure_ascii=False)  # For debugging
      res = get_content(response.json())
      structured_response = self._parse_response(res)
      if((structured_response is not None ) and 
         (self._verify_response(structured_response))):
        return structured_response
    raise ValueError("Failed to get a valid response after maximum attempts")
