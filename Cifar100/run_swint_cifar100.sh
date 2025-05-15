#!/bin/bash

# --- Configuration for Adversarial Evaluation and Training ---
EVAL_FGSM=true
EVAL_PGD=true

# --- Base command arguments (common to all SwinTransformer runs) ---
PYTHON_SCRIPT="Cifar100/SwinT_multi_vector_rama_bernoulli.py"
BASE_ARGS="--epochs 100 --lr 0.01 --batch-size 128 --seed 42 --checkpoint-dir ./checkpoints_cifar100/SwinT"

# --- RAMA specific arguments for SwinTransformer ---
SWINT_RAMA_ARGS="--use-rama --use-normalization --p-value 0.7 --bernoulli-values=0_1 --activation relu"

# --- Constructing Evaluation Flags ---
ADVERSARIAL_EVAL_FLAGS=""
if [ "$EVAL_FGSM" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-fgsm"
fi
if [ "$EVAL_PGD" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-pgd"
fi

# --- Experiment Section: SwinTransformer --- 
echo "-------------------------------------------------"
echo "Starting SwinTransformer Experiments for CIFAR-100"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 1. SwinTransformer Baseline
echo "\nRunning: 1. SwinTransformer Baseline"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 1. SwinTransformer Baseline"
echo "-------------------------------------------------"

# 2. SwinTransformer + AT (Using PGD for Adversarial Training)
AT_MODE_PGD="pgd"
ADVERSARIAL_TRAINING_FLAGS_PGD="--adversarial-training $AT_MODE_PGD"
echo "\nRunning: 2. SwinTransformer + AT (PGD)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 2. SwinTransformer + AT (PGD)"
echo "-------------------------------------------------"

# 3. SwinTransformer + RAMA
echo "\nRunning: 3. SwinTransformer + RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $SWINT_RAMA_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $SWINT_RAMA_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 3. SwinTransformer + RAMA"
echo "-------------------------------------------------"

# 4. SwinTransformer + AT (PGD) + RAMA
echo "\nRunning: 4. SwinTransformer + AT (PGD) + RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $SWINT_RAMA_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $SWINT_RAMA_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 4. SwinTransformer + AT (PGD) + RAMA"
echo "-------------------------------------------------"

echo "\nAll SwinTransformer experiments finished." 