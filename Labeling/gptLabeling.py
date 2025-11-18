from mainClass import LabelingClass


class GptLabeling(LabelingClass):
    def __init__(self):
        super().__init__()
        self.model_id = "openai/gpt-oss-20b-maas"
        self.REGION = "us-central1"
        self.ENDPOINT = f"aiplatform.googleapis.com"


if __name__ == "__main__":
    client = GptLabeling()
    client.labeling("台灣在哪裡？")
