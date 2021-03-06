import os
from typing import TypedDict

from google.cloud import aiplatform
from training_lib.clients import load_labelbox_api_key, load_service_secret

from pipelines.images.classification import ImageSingleClassificationPipeline, ImageMultiClassificationPipeline
from pipelines.images.bounding_box import BoundingBoxPipeline
from pipelines.types import Pipeline
from pipelines.text.ner import NERPipeline
from pipelines.text.classification import TextSingleClassificationPipeline, TextMultiClassificationPipeline
from pipelines.images.custom_classification import ImageKNNClassificationPipeline


LABELBOX_API_KEY = load_labelbox_api_key()
SERVICE_SECRET = load_service_secret()

service_account = os.environ["GOOGLE_SERVICE_ACCOUNT"]
google_cloud_project = os.environ['GOOGLE_PROJECT']
_deployment_name = os.environ['DEPLOYMENT_NAME']
# This should always be set (set in the build arg)

_gcs_bucket = os.environ['GCS_BUCKET']

# Uses default project for your credentials. Can overwrite here if necessary.
aiplatform.init(project=google_cloud_project,
                staging_bucket=f"gs://{_gcs_bucket}")


class Pipelines(TypedDict):
    bounding_box: Pipeline
    ner: Pipeline
    image_single_classification: Pipeline
    image_multi_classification: Pipeline
    text_single_classification: Pipeline
    text_multi_classification: Pipeline
    image_knn_classification: Pipeline

_common_params = [
    _deployment_name, LABELBOX_API_KEY, _gcs_bucket, service_account,
    google_cloud_project
]
pipelines: Pipelines = {
    'bounding_box':
        BoundingBoxPipeline(*_common_params),
    'ner':
        NERPipeline(*_common_params),
    'image_single_classification':
        ImageSingleClassificationPipeline(*_common_params),
    'image_multi_classification':
        ImageMultiClassificationPipeline(*_common_params),
    'text_single_classification':
        TextSingleClassificationPipeline(*_common_params),
    'text_multi_classification':
        TextMultiClassificationPipeline(*_common_params),
    'image_knn_classification':
        ImageKNNClassificationPipeline(*_common_params),
}
