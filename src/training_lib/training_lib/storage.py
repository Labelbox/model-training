import requests
from io import BytesIO
from typing import Tuple, Optional
import logging
import time

from training_lib.errors import InvalidDataRowException

from google.cloud import storage
from google.api_core.retry import Retry
from PIL.Image import Image, open as load_image, DecompressionBombError
import PIL


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 8k x 8k image is too large where downsampling more than that probably won't help.
# This way the user will at least get a meaningful error message. (Instead of a silent oom)
PIL.Image.MAX_IMAGE_PIXELS = 8000**2 // 2

def image_to_bytes(im: Image) -> BytesIO:
    im_bytes = BytesIO()
    im.save(im_bytes, format="jpeg")
    im_bytes.seek(0)
    return im_bytes

@Retry()
def _download_image(image_url: str) -> Image:
    """
    Downloads an image from a url

    Args:
        image_url: A url that references an image

    Returns:
        PIL Image
    """
    return load_image(BytesIO(requests.get(image_url).content))


def get_image_bytes(image_url: str, downsample_factor = 1.) -> Optional[Tuple[Image, Tuple[int,int]]]:
    """
    Fetch image bytes from a url and optionally resize the image.

    Args:
        image_url: A url that references an image
        downsample_factor: How much to scale the image by.
            The new dimensions will be (width * 1/downsample_factor, height * 1/downsample_factor)
    Returns:
        The resized PIL Image
    """
    try:
        with _download_image(image_url) as image:
            w,h = image.size
            with image.resize((int(w *  1./downsample_factor), int(h * 1./downsample_factor))) as resized_image:
                image_bytes = image_to_bytes(resized_image)
                return image_bytes, (w,h)
    except DecompressionBombError:
        raise InvalidDataRowException(f"Image to large : `{image_url}`.")
    except:
        raise InvalidDataRowException(f"Unable to fetch image : `{image_url}`.")


@Retry()
def upload_image_to_gcs(image_bytes: BytesIO, data_row_id: str,
                  bucket: storage.Bucket, dims: Optional[Tuple[int,int]] = None) -> str:
    """
    Uploads images to gcs. Vertex will not work unless the input data is a gcs_uri in a regional bucket hosted in us-central1.
    So we write all labelbox data to this bucket before kicking off the job.

    Args:
        image_bytes: Image bytes
        data_row_id: The id of the image being processed.
        bucket: Cloud storage bucket object
        dims: Optional image dimensions to encode in the filename (used later for reverse etl).

    Returns:
        gcs uri
    """

    if dims is not None:
        # Dims is currently used during inference to scale the prediction
        # When we have media attributes we should use that instead.
        w, h = dims
        suffix = f"_{int(w)}_{int(h)}"
    else:
        suffix = ""
    gcs_key = f"training/images/{data_row_id}{suffix}.jpg"
    blob = bucket.blob(gcs_key)
    blob.upload_from_file(image_bytes, content_type="image/jpg")
    return f"gs://{bucket.name}/{blob.name}"


@Retry()
def upload_text_to_gcs(text_content: str, data_row_id: str,
                       bucket: storage.Bucket) -> str:
    """
    Uploads text content to gcs

    Args:
        text_content: text to upload
        data_row_id: The data row of the content (will be used as the gcs key)
        bucket: Cloud storage bucket object
    Returns:
        gcs uri
    """
    storage.blob._MAX_MULTIPART_SIZE = 1 * 1024 * 1024
    gcs_key = f"training/text/{data_row_id}.txt"
    blob = bucket.blob(gcs_key)
    blob.chunk_size = 1 * 1024 * 1024
    blob.upload_from_string(data=text_content, content_type="text/plain")
    return f"gs://{bucket.name}/{blob.name}"


@Retry()
def upload_ndjson_data(stringified_json : str, bucket: storage.Bucket, gcs_key : str) -> str:
    """
    Uploads ndjson to gcs

    Args:
        stringified_json: ndjson string to write to gcs
        bucket: Cloud storage bucket object
        gcs_key: cloud storage key (basically the file name)
    """
    blob = bucket.blob(gcs_key)
    blob.upload_from_string(stringified_json)
    return f"gs://{bucket.name}/{blob.name}"


def create_gcs_key(job_name: str) -> str:
    """
    Utility for creating a gcs key for etl jobs

    Args:
        job_name: The name of the job (should be named after the etl)
    Returns:
        gcs key for the jsonl file.
    """
    nowgmt = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    return f'etl/{job_name}/{nowgmt}.jsonl'
