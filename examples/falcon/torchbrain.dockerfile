# NDT3 dockerfiles for FALCON challenge
# Smoketest through benchmark start pack ./test_docker_local.sh --docker-name ndt3_smoke
# Submit through EvalAI CLI

# Need devel for flash attn
# FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel # This is the codebase pytorch version, but using updated image for python 3.11
FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-devel

RUN /bin/bash -c "python3 -m pip install falcon_challenge lightning --upgrade"
ENV PREDICTION_PATH="/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL="/tmp/submission.pkl"
ENV GT_PATH="/tmp/ground_truth.pkl"

# Users should install additional decoder-specific dependencies here.
RUN apt-get update && \
    apt-get install -y git

# Copy local codebase and pip install -e .
# Install at root, very safe..
RUN pwd
COPY torch_brain /torch_brain_root/torch_brain/
COPY pyproject.toml /torch_brain_root/pyproject.toml

WORKDIR /torch_brain_root
RUN python3 -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128

WORKDIR /
# Add runfile and necessary context
COPY examples/falcon/configs /configs
ADD examples/falcon/torchbrain_falcon_runner.py decode.py

# Add model checkpoint
ADD "./local_data/dummy.ckpt" data/decoder.pth

# Spec evaluation config
ENV EVALUATION_LOC="local"
# ENV EVALUATION_LOC="remote"

# ENV SPLIT="h1"
# ENV SPLIT="m1"
ENV SPLIT="m2"

ENV BATCH_SIZE=1
ENV PHASE="minival"
# ENV PHASE="test"

# Make sure this matches the mounted data volume path. Generally leave as is.
ENV EVAL_DATA_PATH="/dataset/evaluation_data"

CMD ["/bin/bash", "-c", \
    "python decode.py \
    --evaluation $EVALUATION_LOC \
    --model-path data/decoder.pth \
    --split $SPLIT \
    --batch-size $BATCH_SIZE \
    --phase $PHASE"]