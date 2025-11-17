from dotenv import load_dotenv
import json
import random
import requests
import os
class LabelingClass:
    model_id: str | None = None
    ENDPOINT: str | None = None
    REGION: str | None = None
    
    PROJECT_ID="polstance"
    
    MAX_ATTEMPTS = 3
    return_structure = {chr(ord("a")+i):random.randint(1,3)for i in range(12)}
    return_structure_type = {chr(ord("a")+i): "int" for i in range(12)}
    
    instruction_1 = f"""
你是一個政治立場分析專家，請根據使用者提供的多個標題，分析每個標題的政治立場傾向，並以JSON格式回覆。
接下來會給你多筆資料，請依照每筆資料的內容進行分析，並回傳結果。
若傾向國民黨則標示為\"1\"，若傾向民進黨則標示為\"2\"，若為中立則標示為\"3\"
每篇文章只能有一個傾向，並且僅能夠從上述的三種種類中挑選。
並依照下列的結構幫我輸出結果：
{"\n".join([f"{key}:{value}" for key, value in return_structure_type.items()])}

舉例:
{str(return_structure)}
""" 

    instruction = f"請根據使用者的問題回答"
    
    
    def __init__(self):
        load_dotenv()
        
    def _msg_template(self, prompt):
        return [
            {
              "role": "user",
              "content": prompt
            }
          ]
        
    def _request_config(self, model, msg):
        return {
            "model": model, 
            "stream": False, 
            "max_tokens": 120,
            "messages": msg,
            "system": self.instruction
        }
            
    def _requests_structure(self, config): 
        return {
            "headers":{
                "Authorization": f"Bearer {os.getenv("GCLOUD_KEY")}", 
                "Content-Type": "application/json; charset=utf-8"
                },
            "data":json.dumps(config)
        }

    def _request_url(self):
        return f"https://{self.ENDPOINT}/v1/projects/{self.PROJECT_ID}/locations/{self.REGION}/endpoints/openapi/chat/completions"
        
    def _parse_response(self, response: str) -> dict[str,int] | None:
        for replacement in ["`", " ", "\n", "json"]:
            response = response.replace(replacement, "")
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            return None
    
    def _verify_response(self, response: dict) -> bool:
        for key, value in self.return_structure_type.items():
            if key not in response:
                return False
        return True
    
    def labeling(self, prompt:str) -> dict[str,int]:
        if hasattr(self, '_send_request'):
            structured_response = self._send_request(self._msg_template(prompt))
        else:
            raise NotImplementedError("_send_request method not implemented.")
        return structured_response
    
    def _send_request(self, prompt:list[dict[str,str]]):
        
        req_config = self._request_config(self.model_id, prompt)

        
        attempt = 0
        while attempt < self.MAX_ATTEMPTS:
            response = requests.post(url=self._request_url(),**(self._requests_structure(req_config)))
            with open(f"tmp/{"".join(str(self.model_id).replace("/","-").split('-')[1:4])}_debug.json", "w") as f:
                json.dump(str(response.text), fp=f, indent=4)
            structured_response = self._parse_response(str(response.text))
            if((structured_response is not None ) and True):
                #(self._verify_response(structured_response))):
                return structured_response
            
            attempt += 1
        raise ValueError("Failed to get a valid response after maximum attempts")