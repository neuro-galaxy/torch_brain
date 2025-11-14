#!/usr/bin/env bash

DOCKER_NAME="mysubmission"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

docker run \
    -v $(pwd)/data:/dataset/evaluation_data \
    -e "EVALUATION_LOC=local" \
    --gpus all \
    ${DOCKER_NAME}\