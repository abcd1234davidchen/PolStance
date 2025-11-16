
from google import genai
from google.genai import types
import os
import json
from utils import LabelingClass


class GeminiLabeling(LabelingClass):
    def __init__(self):
        super().__init__()
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
        )

        self.grounding_tool = types.Tool(
            #google_search=types.GoogleSearch()
        )

        self.config = types.GenerateContentConfig(
            tools=[self.grounding_tool],
            response_mime_type="text/plain",
            system_instruction=self.instruction
        )

    
    
    def _send_request(self, prompt):
        if prompt == None:
            raise ValueError("Prompt cannot be None")
        attempt = 0
        #print("gemini prompt:", prompt)
        while attempt < self.MAX_ATTEMPTS:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=self.config,
            )
            with open("tmp/gemini_debug.txt", "w") as f:
                f.write(str(response.text))
            structured_response = self._parse_response(str(response.text))
            if((structured_response is not None ) and 
               (self._verify_response(structured_response))):
                return structured_response
            attempt += 1
        raise ValueError("Failed to get a valid response after maximum attempts")