import base64
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def generateStance(articleText):
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    model = "gemini-flash-latest"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=articleText),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config = types.ThinkingConfig(
            thinking_budget=0,
        ),
        response_mime_type="application/json",
        response_schema=genai.types.Schema(
            type = genai.types.Type.OBJECT,
            required = ["偏向國民黨","偏向民進黨","偏向民眾黨","中立"],
            properties = {
                "偏向國民黨": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "偏向民進黨": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "偏向民眾黨": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
                "中立": genai.types.Schema(
                    type = genai.types.Type.INTEGER,
                ),
            },
        ),
        system_instruction='''你是一個政治立場分析專家，請根據使用者提供的文章內容，分析該文章的政治立場傾向，並以JSON格式回覆，包含以下欄位：偏向國民黨、偏向民進黨、偏向民眾黨、中立。每個欄位的值應為0或1，文章可以有多個傾向，表示該文章對應政治立場的傾向程度。''',
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    print("Wake the fuck up, you have not done anything yet")