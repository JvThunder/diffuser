# Build the Docker Image
docker build -t diffuser .

# Run the Docker Image
docker run -it --rm --gpus all \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$PWD/.d4rl,target=/root/.d4rl \
    diffuser /bin/bash -c "cd /home/code && /bin/bash"

# Train the model
python scripts/train.py --dataset hopper-medium-replay-v2 --logbase logs/cond_a --horizon 32

python scripts/train_mult_horizons.py

# Evaluate the model
python scripts/plan_guided_parallel.py --dataset hopper-medium-replay-v2 --logbase logs/cond_a 

python scripts/eval_mult_weights.py

python scripts/train_eval_mult.py