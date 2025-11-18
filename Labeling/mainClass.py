from dotenv import load_dotenv
import json
import random
import requests
import subprocess


class LabelingClass:
    model_id: str | None = None
    ENDPOINT: str | None = None
    REGION: str | None = None

    PROJECT_ID = "polstance"

    MAX_ATTEMPTS = 3
    return_structure = {chr(ord("a") + i): random.randint(1, 3) for i in range(12)}
    return_structure_type = {chr(ord("a") + i): "int" for i in range(12)}

    instruction = f"""
你是一個政治立場分析專家，並會強烈依照使用者的指示回答問題。現在，請根據使用者提供的多個標題，分析每個標題的政治立場傾向，並以JSON格式回覆。
若傾向國民黨則標示為\"1\"，若傾向民進黨則標示為\"2\"，若為中立則標示為\"3\"
每篇文章只能有一個傾向，並且僅能夠從上述的三種種類中挑選。
並嚴格遵照下列的結構輸出結果：
{"\n".join([f"{key}:{value}" for key, value in return_structure_type.items()])}

舉例:
{str(return_structure)}
"""

    def __init__(self):
        load_dotenv()

    def _msg_template(self, prompt):
        return [
            {"role": "system", "content": self.instruction},
            {"role": "user", "content": prompt},
        ]

    def _request_config(self, model, msg):
        return {
            "model": model,
            "stream": False,
            "thinking_budget": 0,
            "response_format": {"type": "json_object"},
            "messages": msg,
            "system": self.instruction,
            "temperature": 0,
            "max_output_tokens": 1024,
        }

    def _requests_structure(self, config):
        token = (
            subprocess.check_output(["gcloud", "auth", "print-access-token"])
            .decode()
            .strip()
        )
        return {
            "headers": {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            "data": json.dumps(config),
        }

    def _request_url(self):
        return f"https://{self.ENDPOINT}/v1/projects/{self.PROJECT_ID}/locations/{self.REGION}/endpoints/openapi/chat/completions"

    def _parse_response(self, response: str) -> dict[str, int] | None:
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

    def labeling(self, prompt: str) -> dict[str, int]:
        if hasattr(self, "_send_request"):
            structured_response = self._send_request(self._msg_template(prompt))
        else:
            raise NotImplementedError("_send_request method not implemented.")
        return structured_response

    def _send_request(self, prompt: list[dict[str, str]]):
        req_config = self._request_config(self.model_id, prompt)

        attempt = 0
        while attempt < self.MAX_ATTEMPTS:
            response = requests.post(
                url=self._request_url(), **(self._requests_structure(req_config))
            )
            if response.status_code != 200:
                attempt += 1
                #print(f"request failed, retrying.... Error:{response.text}")
                continue
            else:
                response_data = response.json()
                response_text = response.text
            print(response_text)
            with open(
                f"tmp/{''.join(str(self.model_id).replace('/', '-').split('-')[1:4])}_debug.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)

            # 從 API 回應中正確提取模型生成的內容
            model_content = (
                response_data.get("choices", [{}])[0].get("message", {}).get("content")
            )

            structured_response = self._parse_response(model_content)
            if structured_response is not None and self._verify_response(
                structured_response
            ):
                return structured_response
            else:
                #print(response_text)
                attempt += 1
        raise ValueError("Failed to get a valid response after maximum attempts")
