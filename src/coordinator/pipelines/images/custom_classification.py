from typing import Dict, Any
import os
import time
import logging
import json
from google.cloud import aiplatform, storage
from labelbox import ModelRun
from labelbox.data.serialization import NDJsonConverter
import uuid
from pipelines.types import Pipeline, JobStatus, JobState, Job, \
    ClassificationInferenceJob, ClassificationType, PipelineState
from pipelines.images.classification import ImageClassificationPipeline

logger = logging.getLogger("uvicorn")


def parse_uri(gs_uri):
    parts = gs_uri.replace("gs://", "").split("/")
    bucket_name, key = parts[0], "/".join(parts[1:])
    return bucket_name, key


class KNNClassificationTraining(Job):

    def __init__(self, deployment_name: str, gcs_bucket: str,
                 service_account_email: str, google_cloud_project: str):
        self.gcs_bucket = gcs_bucket
        self.service_account_email = service_account_email
        self.google_cloud_project = google_cloud_project
        self.container_name = f"gcr.io/{google_cloud_project}/{deployment_name}/knn_classification_train"

    def run(self, training_file_uri: str, job_name: str) -> JobStatus:
        nowgmt = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        model_file = f'training/knn-image-classification/{nowgmt}.pickle'
        CMDARGS = [
            f"--gcs_bucket={self.gcs_bucket}",
            f"--etl_uri={training_file_uri}",
            f"--model_file={model_file}"
        ]

        job = aiplatform.CustomContainerTrainingJob(
            display_name=job_name,
            container_uri=self.container_name,
        )
        job.run(
            args=CMDARGS,
            service_account=self.service_account_email,
            environment_variables={'GOOGLE_PROJECT': self.google_cloud_project})

        logger.info("model file: %s" % model_file)
        return JobStatus(JobState.SUCCESS,
                         result={
                             'model_file': model_file
                         })


class KNNImageClassificationInference(ClassificationInferenceJob):
    def __init__(self, deployment_name: str, gcs_bucket: str,
                 service_account_email: str, google_cloud_project: str,
                 labelbox_api_key: str):

        self.classification_threshold = 0.2
        super().__init__(labelbox_api_key, 'single', 'image')

        self.gcs_bucket = gcs_bucket
        self.service_account_email = service_account_email
        self.google_cloud_project = google_cloud_project
        self.container_name = f"gcr.io/{google_cloud_project}/" \
                              f"{deployment_name}/knn_classification_predict"

    def run(self, etl_file: str,
            model_run_id: str, model_file: str,
            job_name: str):
        nowgmt = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
        predictions_file = \
            f'inference/knn-image-classification/prediction_knn_{nowgmt}.jsonl'
        CMDARGS = [
            f"--gcs_bucket={self.gcs_bucket}",
            f"--etl_uri={etl_file}",
            f"--model_file={model_file}",
            f"--predictions_file={predictions_file}"
        ]

        job = aiplatform.CustomContainerTrainingJob(
            display_name=job_name,
            container_uri=self.container_name,
        )
        job.run(
            args=CMDARGS,
            service_account=self.service_account_email,
            environment_variables={'GOOGLE_PROJECT': self.google_cloud_project})

        logger.info("predictions file: %s" % predictions_file)

        model_run = self.lb_client._get_single(ModelRun, model_run_id)
        options = self.get_answer_info(model_run.model_id)

        # Since this is not a Vertex model, we need to read the results directly
        # From the jsonl that the inference container created in the bucket
        gcs_client = storage.Client(project=os.environ['GOOGLE_PROJECT'])
        bucket = gcs_client.bucket(self.gcs_bucket)
        preds_blob = bucket.get_blob(predictions_file)
        preds = [json.loads(l) for l in
               preds_blob.download_as_string().decode('UTF-8').split('\n')[:-1]]

        # This is going to be in the same format as annotation_data in the
        # parent class, so after we have it we can procees with all the same
        # methods
        annotation_data = []
        for line in preds:
            annotation_data.append(
                self.build_upload_data(line['prediction'], options,
                                          line['dataRowId']))

        predictions = list(NDJsonConverter.deserialize(annotation_data))
        labels = self.export_model_run_labels(model_run_id, 'image')
        self.compute_metrics(labels, predictions, options)
        upload_task = model_run.add_predictions(
            f'diagnostics-import-{uuid.uuid4()}',
            NDJsonConverter.serialize(predictions))

        upload_task.wait_until_done()
        logger.info(
            f"IMPORT ERRORS : {upload_task.errors}. Imported {len(predictions)}"
        )
        return JobStatus(JobState.SUCCESS)

    def build_upload_data(self, prediction, options, data_row_id):
        return {
            "uuid": str(uuid.uuid4()),
            "answer": {
                'schemaId': options[prediction]['feature_schema_id']
            },
            'dataRow': {
                "id": data_row_id
            },
            "schemaId": options[prediction]['parent_feature_schema_id']
        }


class ImageKNNClassificationPipeline(ImageClassificationPipeline):

    def __init__(self, deployment_name: str,
                 lb_api_key: str,
                 gcs_bucket: str,
                 service_account_email: str,
                 google_cloud_project: str):

        super().__init__('single',
                         deployment_name,
                         lb_api_key,
                         gcs_bucket,
                         service_account_email,
                         google_cloud_project)

        self.training_job = KNNClassificationTraining(deployment_name,
                                                      gcs_bucket,
                                                      service_account_email,
                                                      google_cloud_project)
        self.inference = KNNImageClassificationInference(deployment_name,
                                                         gcs_bucket,
                                                         service_account_email,
                                                         google_cloud_project,
                                                         lb_api_key)

    def run(self, json_data):
        model_run_id, job_name = self.parse_args(json_data)
        self.update_status(PipelineState.PREPARING_DATA, model_run_id)
        etl_status = self.run_job(
            model_run_id, lambda: self.etl_job.run(model_run_id, job_name))

        self.update_status(PipelineState.TRAINING_MODEL,
                           model_run_id,
                           metadata={'training_data_input': etl_status.result['etl_file']})

        training_status = self.run_job(
            model_run_id,
            lambda: self.training_job.run(etl_status.result['etl_file'], job_name))

        self.update_status(
            PipelineState.TRAINING_MODEL,
            model_run_id,
            metadata={'model_id': training_status.result['model_file']})

        self.run_job(
            model_run_id, lambda: self.inference.run(
                etl_status.result['etl_file'], model_run_id, training_status.result[
                    'model_file'], job_name))
        self.update_status(PipelineState.COMPLETE, model_run_id)