#!/bin/bash
IMAGE_NAME=kedro-app

docker build --no-cache -t $IMAGE_NAME -f Dockerfile-kedro .

# Mount data so that local data folder is changed
docker run --rm \
    -v "$(pwd)/data:/app/data" \
    $IMAGE_NAME
