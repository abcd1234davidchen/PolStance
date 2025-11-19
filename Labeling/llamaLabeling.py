from mainClass import LabelingClass
import traceback


class LlamaLabeling(LabelingClass):
    def __init__(self):
        super().__init__()
        self.model_id = "meta/llama-4-maverick-17b-128e-instruct-maas"
        self.REGION = "us-east5"
        self.ENDPOINT = f"us-east5-aiplatform.googleapis.com"

    def _get_response_text(self, response: dict) -> str:
        try:
            res = response["choices"][0]["message"]["content"]
            res = res.replace("```json", "").replace("```", "").replace('"', '"')
        except (KeyError, IndexError):
            res = ""
            print(f"Warning: Unexpected response structure. {traceback.format_exc()}")
        return res


if __name__ == "__main__":
    client = LlamaLabeling()
    client.labeling("台灣在哪裡？")
