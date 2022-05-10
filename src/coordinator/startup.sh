#!/bin/bash

create_cert () {
    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -sha256 -days 180 -subj "/C=US/ST=California/L=San Fransisco/O=Labelbox/OU=Org/CN=www.labelbox.com" -nodes
}
create_cert
uvicorn "coordinator:app" --host=0.0.0.0 --port=443 --reload --reload-include="*.pem" --ssl-keyfile="key.pem" --ssl-certfile="cert.pem" &
while true
do
  sleep 14688000 # 170 days in seconds gives a 10 day buffer
  while : ; do
    JOB_COUNT=$(curl -k -s https://0.0.0.0/job_count)
    [[ ! $JOB_COUNT -eq 0 ]] || break
    sleep 3600 # try once an hour to see if there are any active jobs.
    echo "Waiting for jobs to finish. There are $JOB_COUNT jobs currently running"
  done
  create_cert
 echo "New cert created!"
done