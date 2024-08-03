# Build the Docker Image
docker build -t diffuser .

# Run the Docker Image
docker run -it --rm --gpus all \
    --mount type=bind,source=$PWD,target=/home/code \
    --mount type=bind,source=$PWD/.d4rl,target=/root/.d4rl \
    diffuser /bin/bash -c "cd /home/code && /bin/bash"

# Train the model
python scripts/train.py --dataset halfcheetah-expert-v2 --logbase logs/cond_a_film --film

python scripts/train_mult.py

# Evaluate the model
python scripts/plan_guided_parallel.py --dataset hopper-medium-replay-v2 --logbase logs/cond_a_film --horizon 32 --guidance_weight 1.5 --m_temp -1.0 --film

python scripts/eval_mult.py

python scripts/plan_guided_parallel.py --dataset walker2d-medium-v2 --logbase logs/av_cond_nofilm --render

python scripts/train_mult.py && python scripts/eval_mult.py