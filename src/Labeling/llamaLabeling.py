from Labeling.mainClass import LabelingClass
import traceback
from typing import Any


class LlamaLabeling(LabelingClass):
    def __init__(self):
        super().__init__()
        self.model_id = "meta-llama/llama-4-maverick"

    def _request_config(self, model, msg) -> dict[str, Any]:
        return {
            "model": model,
            "messages": msg,
            "provider": {
                "allow_fallbacks": True,
                "only": [
                    "Groq",
                    "friendli",
                    "deepinfra/base"
                ]
            }
        }
    
if __name__ == "__main__":
    client = LlamaLabeling()
    client.labeling("台灣在哪裡？")
