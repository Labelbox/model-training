import requests
import hmac
import hashlib
import json
"""
This script simulates a webhook event.

# Make sure that the LABELBOX_API_KEY cooresponds to the org that belongs to this project.
# If you want any real data to be produced, there should be some bounding boes in the project.
"""
model_run_id = "9e13ae3f-894e-0d86-1686-2ea032302bda"  #"9da8885e-1d6c-0fc6-765b-c05c5166a70e"
secret = b'test_secret'

payload = json.dumps({
    'modelRunId': model_run_id,
    'modelType': 'image_knn_classification'
})
signature = "sha1=" + hmac.new(
    secret, msg=payload.encode(), digestmod=hashlib.sha1).hexdigest()
res = requests.post("https://0.0.0.0/model_run",
                    data=payload,
                    headers={'X-Hub-Signature': signature})
