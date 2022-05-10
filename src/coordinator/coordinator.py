from typing import Dict, Any
import hashlib
import hmac
import json
import logging
import datetime

from fastapi import BackgroundTasks, FastAPI, HTTPException, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.logger import logger
import uvicorn
from time import sleep

from config import pipelines, SERVICE_SECRET
from pipelines.types import PipelineState

logger = logging.getLogger("uvicorn")
logger.setLevel(logging.DEBUG)
app = FastAPI()


in_progress = set()

async def run(json_data: Dict[str, Any]):
    job_name = json_data['job_name']
    if job_name in in_progress:
        raise Exception(f"Job `{job_name}` already exists")

    in_progress.add(job_name)

    try:
        pipeline = json_data['pipeline']
        await run_in_threadpool(pipelines[pipeline].run, json_data)
    except Exception as e:
        pipelines[pipeline].update_status(PipelineState.FAILED,
                                          json_data['model_run_id'],
                                          error_message=str(e))
    finally:
        in_progress.remove(job_name)


@app.get("/job_count")
async def job_count():
    return len(in_progress)


@app.get("/models")
async def models(X_Hub_Signature: str = Header(None)):
    computed_signature = hmac.new(SERVICE_SECRET.encode(),
                                  digestmod=hashlib.sha1).hexdigest()
    if X_Hub_Signature != "sha1=" + computed_signature:
        raise HTTPException(
            status_code=401,
            detail=
            "Error: computed_signature does not match signature provided in the headers"
        )
    return {k: {} for k in pipelines.keys()}


@app.post("/model_run")
async def model_run(request: Request,
                    background_tasks: BackgroundTasks,
                    X_Hub_Signature: str = Header(None)):
    req = await request.body()
    computed_signature = hmac.new(SERVICE_SECRET.encode(),
                                  msg=req,
                                  digestmod=hashlib.sha1).hexdigest()
    if X_Hub_Signature != "sha1=" + computed_signature:
        raise HTTPException(
            status_code=401,
            detail=
            "Error: computed_signature does not match signature provided in the headers"
        )

    data = json.loads(req.decode("utf8"))
    logger.info(f"Training job request recieved. Params: {data}")
    validate_payload(data)
    background_tasks.add_task(run, data)


def validate_payload(data: Dict[str, str]):
    valid_pipelines = list(pipelines.keys())
    if 'modelType' not in data:
        raise KeyError(
            "Must provide `modelType` key indicating which pipeline to run. "
            f"Should be one of: {valid_pipelines}")
    if not (data['modelType'] in list(pipelines.keys())):
        raise ValueError(
            f"Unkonwn pipeline `{data['modelType']}`. Expected one of {valid_pipelines}"
        )
    data['pipeline'] = data['modelType']

    if 'modelRunId' not in data:
        raise KeyError("Must provide `modelRunId`")

    data['model_run_id'] = data['modelRunId']

    if 'job_name' not in data:
        data[
            'job_name'] = f'{data["pipeline"]}_{str(datetime.datetime.now()).replace(" ", "_")}'
    pipelines[data['pipeline']].parse_args(data)


@app.get("/ping")
def health_check():
    return "pong"

