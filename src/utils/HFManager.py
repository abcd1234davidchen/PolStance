from huggingface_hub import hf_api, hf_hub_download
import os
from dotenv import load_dotenv
from utils.DBManager import DBManager
load_dotenv()


class HFManager:
    def __init__(self, access_type="dataset") -> None:
        if access_type == "dataset":
            self.repo_id = "TWCKaijin/PolStance"
            self.filename = f"{os.getenv("DBNAME")}.db"
            self.access_type = "dataset"
            self.token = os.getenv("DATASET_KEY")
        elif access_type == "model":
            self.repo_id = "abcd1234davidchen/PolStanceBERT"
            self.filename = f"{os.getenv("MODEL_NAME")}.pth"
            self.access_type = "model"
            self.token = os.getenv("MODEL_KEY")
        self.db_manager = None
        self.db_path = None
        self.hf_api = hf_api.HfApi(token=self.token)

    def download_db(self) -> DBManager:
        sql_file = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            repo_type=self.access_type,
            token=self.token,
            local_dir=os.getcwd(),
        )
        self.db_path = sql_file
        print(f"Downloaded database to {sql_file}")
        self.db_manager = DBManager(sql_file)
        return self.db_manager

    def upload_db(self, commit_message="Update automatcally") -> None:
        if self.db_manager:
            self.db_manager.close()

        if self.db_path and os.path.exists(self.db_path):
            self.hf_api.upload_file(
                path_or_fileobj=self.db_path,
                path_in_repo=self.filename,
                repo_id=self.repo_id,
                repo_type=self.access_type,
                commit_message=commit_message,
            )
            print(f"Successfully uploaded {self.filename} to {self.repo_id}")
        else:
            print("No database file to upload")
    
    def upload_model(self,commit_message="Update model automatically") -> None:
        if os.path.exists(self.filename):
            self.hf_api.upload_file(
                path_or_fileobj=self.filename,
                path_in_repo=self.filename,
                repo_id=self.repo_id,
                repo_type=self.access_type,
                commit_message=commit_message,
            )
            print(f"Successfully uploaded {self.filename} to {self.repo_id}")
        else:
            print("No model file to upload")
    
    def download_model(self):
        model_file = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            repo_type=self.access_type,
            token=self.token,
            local_dir=os.getcwd(),
        )
        print(f"Downloaded model to {model_file}")
        return model_file
    
if __name__ == "__main__":
    pass
    # hf_manager = HFManager(access_type="dataset")
    # db_manager = hf_manager.download_db()
    # hf_manager.upload_db(commit_message="Updated database from HFManager")
    # hf_model_manager = HFManager(access_type="model")
    # model_path = hf_model_manager.download_model()
    # hf_model_manager.upload_model(commit_message="Updated model from HFManager")