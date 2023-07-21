"""Unit tests for SHIVA preprocessing tools"""

import os
import json

import urllib3
from minio import Minio
cert_path = os.path.join(
                    os.path.dirname(__file__), '..', 'share',
                    'randd_cacert.pem')
print(cert_path)
os.path.exists(cert_path)
httpClient = urllib3.PoolManager(
                cert_reqs='CERT_REQUIRED',
                ca_certs=cert_path
     )

secrets = json.load(os.path.join(__file__, '..', '.secrets',
                                 'credentials.json'))

minio_client = Minio(endpoint='minio.swomed2dev01.cadesis:9000',
                     access_key='aWMUOPTIRpGSRnae',
                     secret_key='a2BZjWVyn5FVFzbVQ6ekkEGfiqMNHFp8',
                     secure=False,
                     http_client=httpClient
                     )

req = minio_client.fget_object(bucket_name="unittest-data",
                               object_name="01.nii",
                               file_path="01.nii")


print(type(req))
