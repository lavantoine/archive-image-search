import boto3
from botocore.client import Config
import streamlit as st
from pathlib import Path
from io import BytesIO

class S3():
    def __init__(self) -> None:
        self.client = boto3.client(
            's3',
            aws_access_key_id=st.secrets['OVH']['ACCESS_KEY_ID'],
            aws_secret_access_key=st.secrets['OVH']['SECRET_ACCESS_KEY'],
            endpoint_url=st.secrets['OVH']['ENDPOINT'],
            config=Config(signature_version='s3v4')  # works well with OVH
        )
        self.bucket = 'images-mae'
    
    def upload_file(self, filepath: Path) -> None:
        try:
            self.client.upload_file(filepath, bucket, filepath.name)
        except Exception as e:
            print(f'Error while uploading file \"{filepath.name}\": {e}')
    
    def download_file(self, filename) -> BytesIO | None:
        try:
            file_obj = BytesIO()
            self.client.download_fileobj(self.bucket, filename, file_obj)
            file_obj.seek(0)
            return file_obj
        except Exception as e:
            print(f'Error while retrieving file \"{filename}\": {e}')
    
    def file_exists(self, filename):
        try:
            self.client.head_object(self.bucket, filename)
            return True
        except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False
                else:
                    raise
    
    def list_buckets(self):
        return [bucket['Name'] for bucket in self.client.list_buckets()]