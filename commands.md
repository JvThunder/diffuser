# Build the Docker Image
docker build -t diffuser .

# Run the Docker Image
docker run -it --rm --gpus all \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$PWD/.d4rl,target=/root/.d4rl \
    diffuser /bin/bash -c "cd /home/code && /bin/bash"

# Train the model
python scripts/train.py --dataset walker2d-medium-replay-v2 --logbase logs
OR
python scripts/train_inverse.py --dataset walker2d-medium-replay-v2 --logbase logs

# Evaluate the model
python scripts/plan_guided.py --dataset walker2d-medium-replay-v2 --logbase logs

# List of available datasets
- halfcheetah-medium-expert-v2
- hopper-medium-replay-v2
- walker2d-medium-replay-v2