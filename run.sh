#!/bin/bash
IMAGE_NAME=kedro-app

docker build -t $IMAGE_NAME -f Dockerfile-kedro .

# Mount data so that local data folder is changed
docker run --rm \
    -u $(id -u):$(id -g) \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/saved_models:/app/saved_models" \
    "$IMAGE_NAME"

