from utils import LabelingClass



class LlamaLabeling(LabelingClass):
	def __init__(self):
		super().__init__()
		self.model_id = "meta/llama-4-maverick-17b-128e-instruct-maas"
		self.REGION= "us-east5"
		self.ENDPOINT=f"us-east5-aiplatform.googleapis.com"
		


if __name__ == '__main__':
	client = LlamaLabeling()
	client.labeling("台灣在哪裡？")
    
    

  