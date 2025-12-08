from Labeling.mainClass import LabelingClass
import traceback
from typing import Any

class GeminiLabeling(LabelingClass):
    def __init__(self):
        super().__init__()
        self.model_id = "google/gemini-2.5-flash"


    def _request_config(self, model, msg) -> dict[str, Any]:
        return {

            "model": model,
            "messages": msg,
            "reasoning": {
                "effort": "minimal" 
            }
        }

if __name__ == "__main__":
    client = GeminiLabeling()
    client.labeling("台灣在哪裡？")
