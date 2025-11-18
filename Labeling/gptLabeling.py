from mainClass import LabelingClass
import traceback

class GptLabeling(LabelingClass):
    def __init__(self):
        super().__init__()
        self.model_id = "openai/gpt-oss-20b-maas"
        self.REGION = "us-central1"
        self.ENDPOINT = f"aiplatform.googleapis.com"

    def _get_response_text(self, response: dict) -> str:
        try:
            res = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            res = ""
            print(f"Warning: Unexpected response structure. {traceback.format_exc()}")
        return res
    
if __name__ == "__main__":
    client = GptLabeling()
    client.labeling("台灣在哪裡？")
