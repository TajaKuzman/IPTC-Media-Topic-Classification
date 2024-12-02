#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

echo "GPU:"
echo ${CUDA_VISIBLE_DEVICES}

echo "5k_sample_1"

python -u machine-learning-experiments-code/train_and_eval_model.py "5k-model-sample_1-v1" "5k_sample_1" > results/5k_model_sample_1_v1_log.md

echo "5k_sample_2"

python -u machine-learning-experiments-code/train_and_eval_model.py "5k-model-sample_2-v1" "5k_sample_2" > results/5k_model_sample_2_v1_log.md

echo "2.5k_sample_2"

python -u machine-learning-experiments-code/train_and_eval_model.py "2.5k-model-sample_2-v1" "2.5k_sample_2" > results/2.5k_model_sample_2_v1_log.md

echo "1k_sample_1"

python -u machine-learning-experiments-code/train_and_eval_model.py "1k-model-sample_1-v1" "1k_sample_1" > results/1k_model_sample_1_v1_log.md

echo "1k_sample_2"

python -u machine-learning-experiments-code/train_and_eval_model.py "1k-model-sample_2-v1" "1k_sample_2" > results/1k_model_sample_2_v1_log.md