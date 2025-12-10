import boto3
from botocore.exceptions import ClientError
from config import Config


class S3Storage:
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=Config.AWS_ACCESS_KEY,
            aws_secret_access_key=Config.AWS_SECRET_KEY,
        )
        self.bucket_name = Config.AWS_BUCKET_NAME

    def upload_file(self, file_obj=None, object_name=None, filename=None):
        if file_obj is None:
            file_obj = open(filename, "rb")
        if object_name is None:
            object_name = filename
        try:
            self.s3_client.upload_fileobj(file_obj, self.bucket_name, object_name)
        except ClientError as e:
            print(f"Error uploading file: {e}")
            return False
        return True

    def get_file(self, object_name=None, filename=None):
        if object_name is None:
            object_name = filename
        try:
            self.s3_client.download_file(self.bucket_name, object_name, filename)
        except ClientError as e:
            print(f"Error downloading file: {e}")
            return False
        return True

