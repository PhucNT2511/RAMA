#!/bin/bash

# --- Configuration for Adversarial Evaluation and Training ---
EVAL_FGSM=true
EVAL_PGD=true
AT_MODE_PGD="--at-attack pgd --at-epsilon 0.03 --at-alpha 0.01 --at-iter 7" # AT PGD specific params

# --- Base command arguments (common to all Swin-T runs on Tiny ImageNet) ---
BASE_ARGS="--epochs 100 --lr 0.001 --batch-size 128 --seed 42 --checkpoint-dir ./checkpoints_tiny_imagenet/SwinT"
DATASET_ARGS="--data-dir ./data_tiny_imagenet"

# --- Constructing Evaluation Flags ---
ADVERSARIAL_EVAL_FLAGS=""
if [ "$EVAL_FGSM" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-fgsm"
fi
if [ "$EVAL_PGD" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-pgd"
fi

# --- Adversarial Training Flags ---
ADVERSARIAL_TRAINING_FLAGS_PGD="--adversarial-training $AT_MODE_PGD"

# ==============================================================================
# Swin-T Bernoulli RAMA Experiments for Tiny ImageNet
# ==============================================================================
PYTHON_SCRIPT_BERNOULLI="Tiny_Imagenet/SwinT_multi_vector_rama_bernoulli.py"
SWINT_RAMA_BERNOULLI_ARGS="--use-rama --use-normalization --p-value 0.7 --bernoulli-values=0_1 --activation silu --sqrt-dim False"

echo "-------------------------------------------------"
echo "Starting Swin-T (Bernoulli RAMA) Experiments for Tiny ImageNet"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 1. Swin-T Baseline (Bernoulli script, no RAMA)
echo "\nRunning: 1. Swin-T Baseline (Bernoulli Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 1. Swin-T Baseline (Bernoulli Script)"
echo "-------------------------------------------------"

# 2. Swin-T + AT (PGD) (Bernoulli script, no RAMA)
echo "\nRunning: 2. Swin-T + AT (PGD) (Bernoulli Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 2. Swin-T + AT (PGD) (Bernoulli Script)"
echo "-------------------------------------------------"

# 3. Swin-T + Bernoulli RAMA
echo "\nRunning: 3. Swin-T + Bernoulli RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $SWINT_RAMA_BERNOULLI_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $SWINT_RAMA_BERNOULLI_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 3. Swin-T + Bernoulli RAMA"
echo "-------------------------------------------------"

# 4. Swin-T + Bernoulli RAMA + AT (PGD)
echo "\nRunning: 4. Swin-T + Bernoulli RAMA + AT (PGD)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $SWINT_RAMA_BERNOULLI_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $SWINT_RAMA_BERNOULLI_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 4. Swin-T + Bernoulli RAMA + AT (PGD)"
echo "-------------------------------------------------"

# ==============================================================================
# Swin-T Gaussian RAMA Experiments for Tiny ImageNet
# ==============================================================================
PYTHON_SCRIPT_GAUSSIAN="Tiny_Imagenet/SwinT_multi_vector_rama_gaussian.py"
SWINT_RAMA_GAUSSIAN_ARGS="--use-rama --use-normalization --lambda-value 0.2 --activation relu --sqrt-dim False --sigma-p-value 1.0"

echo "\n\n-------------------------------------------------"
echo "Starting Swin-T (Gaussian RAMA) Experiments for Tiny ImageNet"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 5. Swin-T Baseline (Gaussian script, no RAMA)
echo "\nRunning: 5. Swin-T Baseline (Gaussian Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 5. Swin-T Baseline (Gaussian Script)"
echo "-------------------------------------------------"

# 6. Swin-T + AT (PGD) (Gaussian script, no RAMA)
echo "\nRunning: 6. Swin-T + AT (PGD) (Gaussian Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 6. Swin-T + AT (PGD) (Gaussian Script)"
echo "-------------------------------------------------"

# 7. Swin-T + Gaussian RAMA
echo "\nRunning: 7. Swin-T + Gaussian RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $SWINT_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $SWINT_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 7. Swin-T + Gaussian RAMA"
echo "-------------------------------------------------"

# 8. Swin-T + Gaussian RAMA + AT (PGD)
echo "\nRunning: 8. Swin-T + Gaussian RAMA + AT (PGD)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $SWINT_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $SWINT_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 8. Swin-T + Gaussian RAMA + AT (PGD)"
echo "-------------------------------------------------"

echo "\nAll Swin-T Tiny ImageNet experiments finished." 