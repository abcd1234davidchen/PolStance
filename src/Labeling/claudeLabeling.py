from Labeling.mainClass import LabelingClass
import traceback
from typing import Any


class ClaudeLabeling(LabelingClass):
    def __init__(self):
        super().__init__()
        self.model_id = "anthropic/claude-haiku-4.5"

    def _request_config(self, model, msg) -> dict[str, Any]:
        return {
            "model": model,
            "messages": msg,
            "provider": {
                "allow_fallbacks": True
            },
            "reasoning": {
                "effort": "minimal" 
            }
        }
    
if __name__ == "__main__":
    client = ClaudeLabeling()
    client.labeling("台灣在哪裡？")
