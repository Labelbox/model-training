
import json
import argparse
import logging
from typing import Optional

from training_lib.errors import InvalidLabelException
from training_lib.storage import get_image_bytes, upload_image_to_gcs
from training_lib.etl import process_labels_in_threadpool, get_labels_for_model_run, partition_mapping, validate_label
from training_lib.clients import get_lb_client, get_gcs_client
from training_lib.storage import upload_ndjson_data, create_gcs_key

from labelbox import Client
from labelbox.data.annotation_types import Label, Rectangle
from google.cloud.storage.bucket import Bucket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


VERTEX_MIN_BBOX_DIM = 9
VERTEX_MAX_EXAMPLES_PER_IMAGE = 500
VERTEX_MIN_TRAINING_EXAMPLES = 50


def clip_and_round(value: float) -> float:
    """
    Constrains the value to be between 0 and 1 (otherwise vertex will filter out these examples).
    Args:
        value: number to clip and round
    Returns:
        rounded number clipped to be between 0 and 1
    """

    rounded = round(value, 2)
    return max(min(rounded, 1), 0)


def convert_to_percent(bbox: Rectangle, image_w, image_h):
    """
    Converts a labelbox bounding box annotation into a vertex bounding box annotation.
    Args:
          bbox: Bounding box annotation to convert
          image_w: image width to scale by
          image_h: image height to scale by
    """
    return {
        "xMin": clip_and_round(bbox.start.x / image_w),
        "yMin": clip_and_round(bbox.start.y / image_h),
        "xMax": clip_and_round(bbox.end.x / image_w),
        "yMax": clip_and_round(bbox.end.y / image_h),
    }


def process_label(label: Label, bucket: Bucket, downsample_factor: int = 4) -> Optional[str]:
    """
    Function for converting a labelbox Label object into a vertex json label for object detection.
    Args:
        label: the label to convert
    Returns:
        Stringified json representing a vertex label
    """
    bounding_box_annotations = []
    validate_label(label)
    image_bytes, (w,h) = get_image_bytes(label.data.url)

    for annotation in label.annotations:
        if isinstance(annotation.value, Rectangle):
            bbox = annotation.value
            if (bbox.end.x - bbox.start.x) < (
                    VERTEX_MIN_BBOX_DIM * downsample_factor) or (
                        bbox.end.y - bbox.start.y) < (VERTEX_MIN_BBOX_DIM *
                                                      downsample_factor):

                new_x = round((bbox.end.x - bbox.start.x) * 1./downsample_factor)
                new_y = round((bbox.end.y - bbox.start.y) * 1./downsample_factor)
                logger.warning(
                    f"Resized bounding box is too small ({new_x}, {new_y})."
                )
                continue

            bounding_box_annotations.append({
                "displayName": annotation.name,
                **convert_to_percent(bbox, w, h)
            })

    if len(bounding_box_annotations) == 0:
        raise InvalidLabelException(f"There are 0 valid annotations for data row `{label.data.uid}`.")

    gcs_uri = upload_image_to_gcs(image_bytes, label.data.uid, bucket, (w, h))
    return json.dumps({
        'imageGcsUri':
            gcs_uri,
        'boundingBoxAnnotations':
            bounding_box_annotations[:VERTEX_MAX_EXAMPLES_PER_IMAGE],
        'dataItemResourceLabels': {
            "aiplatform.googleapis.com/ml_use": partition_mapping[label.extra.get("Data Split")],
            "dataRowId": label.data.uid
        }
    })


def bounding_box_etl(lb_client: Client, model_run_id: str, bucket) -> str:
    """
    Creates a jsonl file that is used for input into a vertex ai training job

    This code barely validates the requirements as listed in the vertex documentation.
    Read more about the restrictions here:
        - https://cloud.google.com/vertex-ai/docs/datasets/prepare-image#object-detection

    Args:
        lb_client: Labelbox client object
        model_run_id: the id of the model run to export labels from
        bucket: Cloud storage bucket used to upload image data to
    Retuns:
        stringified ndjson
    """

    labels = get_labels_for_model_run(lb_client, model_run_id)
    training_data = process_labels_in_threadpool(process_label, labels, bucket)

    if len(training_data) < VERTEX_MIN_TRAINING_EXAMPLES:
        raise InvalidLabelException("Not enough training examples provided")

    return "\n".join(training_data)


def main(model_run_id: str, gcs_bucket: str, gcs_key: Optional[str] = None):
    lb_client = get_lb_client()
    bucket = get_gcs_client().bucket(gcs_bucket)
    json_data = bounding_box_etl(lb_client, model_run_id, bucket)
    gcs_key = gcs_key or create_gcs_key('bounding-box')
    etl_file = upload_ndjson_data(json_data, bucket, gcs_key)
    logger.info("ETL Complete. URI: %s", f"{etl_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Vertex AI ETL Runner')
    parser.add_argument('--gcs_bucket', type=str, required=True)
    parser.add_argument('--model_run_id', type=str, required=True)
    parser.add_argument('--gcs_key', type=str, required=False, default=None)
    args = parser.parse_args()
    main(**vars(args))
