#!/bin/bash

# --- Configuration for Adversarial Evaluation and Training ---
EVAL_FGSM=true
EVAL_PGD=true

# --- Base command arguments (common to all ResNet-18 runs) ---
PYTHON_SCRIPT="Cifar100/Resnet_multi_vector_rama_bernoulli.py"
BASE_ARGS="--epochs 100 --lr 0.01 --batch-size 128 --seed 42 --checkpoint-dir ./checkpoints_cifar100/ResNet18"

# --- RAMA specific arguments for ResNet-18 ---
RESNET_RAMA_ARGS="--use-rama --use-normalization --p-value 0.7 --bernoulli-values=0_1 --activation relu"

# --- Constructing Evaluation Flags ---
ADVERSARIAL_EVAL_FLAGS=""
if [ "$EVAL_FGSM" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-fgsm"
fi
if [ "$EVAL_PGD" = true ] ; then
    ADVERSARIAL_EVAL_FLAGS+=" --eval-pgd"
fi

# --- Experiment Section: ResNet-18 --- 
echo "-------------------------------------------------"
echo "Starting ResNet-18 Experiments for CIFAR-100"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 1. ResNet-18 Baseline
echo "\nRunning: 1. ResNet-18 Baseline"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 1. ResNet-18 Baseline"
echo "-------------------------------------------------"

# 2. ResNet-18 + AT (Using PGD for Adversarial Training)
AT_MODE_PGD="pgd"
ADVERSARIAL_TRAINING_FLAGS_PGD="--adversarial-training $AT_MODE_PGD"
echo "\nRunning: 2. ResNet-18 + AT (PGD)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 2. ResNet-18 + AT (PGD)"
echo "-------------------------------------------------"

# 3. ResNet-18 + RAMA
echo "\nRunning: 3. ResNet-18 + RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $RESNET_RAMA_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $RESNET_RAMA_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 3. ResNet-18 + RAMA"
echo "-------------------------------------------------"

# 4. ResNet-18 + AT (PGD) + RAMA
echo "\nRunning: 4. ResNet-18 + AT (PGD) + RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $RESNET_RAMA_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT $BASE_ARGS $RESNET_RAMA_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 4. ResNet-18 + AT (PGD) + RAMA"
echo "-------------------------------------------------"

echo "\nAll ResNet-18 experiments finished." 