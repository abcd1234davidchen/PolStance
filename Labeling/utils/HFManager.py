from huggingface_hub import hf_api, hf_hub_download
import os
from dotenv import load_dotenv
from DBManager import DbManager

load_dotenv()


class HFManager:
    def __init__(self, filename="article.db") -> None:
        self.repo_id = "TWCKaijin/PolStance"
        self.filename = filename
        self.token = os.getenv("DATASET_KEY")
        self.db_manager = None
        self.db_path = None
        self.hf_api = hf_api.HfApi(token=self.token)

    def download_db(self) -> DbManager:
        sql_file = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            repo_type="dataset",
            token=self.token,
        )
        self.db_path = sql_file
        self.db_manager = DbManager(sql_file)
        return self.db_manager

    def upload_db(self, commit_message="Update automately") -> None:
        if self.db_manager:
            self.db_manager.close()

        if self.db_path and os.path.exists(self.db_path):
            self.hf_api.upload_file(
                path_or_fileobj=self.db_path,
                path_in_repo=self.filename,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=commit_message,
            )
            print(f"Successfully uploaded {self.filename} to {self.repo_id}")
        else:
            print("No database file to upload")
