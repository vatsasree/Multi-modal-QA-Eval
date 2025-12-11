#!/bin/bash

# Configuration
IMAGE_NAME="patram_155"
CONTAINER_NAME="hf_model_155"
DOCKERFILE_PATH="/projects/data/vision-team/shanmukha_sreevatsa/Patram/Dockerfile"
LOG_FILE="docker_setup.log"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to exec into container
attach_container() {
    log "Attaching to container ${CONTAINER_NAME}..."
    exec docker exec -it ${CONTAINER_NAME} bash
}

# Check if image exists
if ! docker images --format '{{.Repository}}' | grep -q "^${IMAGE_NAME}$"; then
    log "Image ${IMAGE_NAME} not found. Building with UID/GID..."
    docker build -t ${IMAGE_NAME} \
        -f ${DOCKERFILE_PATH} \
        --build-arg UID=$(id -u) \
        --build-arg GID=$(id -g) \
        $(dirname ${DOCKERFILE_PATH}) | tee -a "$LOG_FILE"
else
    log "Image ${IMAGE_NAME} already exists."
fi

# Check if container exists
if ! docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    log "Container ${CONTAINER_NAME} not found. Creating and starting..."
    docker run -dit --name ${CONTAINER_NAME} \
        --cpus 128 \
        --gpus all \
        --network=host \
        -v /projects/data/vision-team/hrithik_sagar/models_inferences:/projects/data/vision-team/hrithik_sagar/models_inferences \
        -v /projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/:/projects/data/vision-team/venkat_kesav/GR_Model_Training_with_Swift/ \
        -v /home/shanmukha_sreevatsa/.ssh:/home/shanmukha_sreevatsa/.ssh \
        -v /projects/data/vision-team/shanmukha_sreevatsa/hf_cache:/projects/data/vision-team/shanmukha_sreevatsa/hf_cache \
        -v /projects/data/vision-team/shanmukha_sreevatsa/environments:/home/shanmukha_sreevatsa/environments/ \
        -v /projects/data/vision-team/:/projects/data/vision-team/ \
        ${IMAGE_NAME} | tee -a "$LOG_FILE"
    attach_container
else
    if [ "$(docker inspect -f '{{.State.Running}}' ${CONTAINER_NAME})" = "true" ]; then
        log "Container ${CONTAINER_NAME} is already running."
        attach_container
    else
        log "Container ${CONTAINER_NAME} exists but is stopped. Starting..."
        docker start ${CONTAINER_NAME} | tee -a "$LOG_FILE"
        attach_container
    fi
fi
