#!/bin/bash

# --- Configuration for Adversarial Evaluation and Training ---
EVAL_FGSM=true
EVAL_PGD=true
AT_MODE_PGD="--at-attack pgd --at-epsilon 0.03 --at-alpha 0.01 --at-iter 7" # AT PGD specific params

# --- Base command arguments (common to all ResNet-18 runs on Tiny ImageNet) ---
BASE_ARGS="--epochs 100 --lr 0.01 --batch-size 128 --seed 42 --checkpoint-dir ./checkpoints_tiny_imagenet/ResNet18"
DATASET_ARGS="--data-dir ./data_tiny_imagenet" # Specify data directory for Tiny ImageNet

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
# ResNet-18 Bernoulli RAMA Experiments for Tiny ImageNet
# ==============================================================================
PYTHON_SCRIPT_BERNOULLI="Tiny_Imagenet/Resnet_multi_vector_rama_bernoulli.py"
RESNET_RAMA_BERNOULLI_ARGS="--use-rama --use-normalization --p-value 0.8 --bernoulli-values=0_1 --activation relu --sqrt-dim False"

echo "-------------------------------------------------"
echo "Starting ResNet-18 (Bernoulli RAMA) Experiments for Tiny ImageNet"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 1. ResNet-18 Baseline (Bernoulli script, no RAMA)
echo "\nRunning: 1. ResNet-18 Baseline (Bernoulli Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 1. ResNet-18 Baseline (Bernoulli Script)"
echo "-------------------------------------------------"

# 2. ResNet-18 + AT (PGD) (Bernoulli script, no RAMA)
echo "\nRunning: 2. ResNet-18 + AT (PGD) (Bernoulli Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 2. ResNet-18 + AT (PGD) (Bernoulli Script)"
echo "-------------------------------------------------"

# 3. ResNet-18 + Bernoulli RAMA
echo "\nRunning: 3. ResNet-18 + Bernoulli RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $RESNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $RESNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 3. ResNet-18 + Bernoulli RAMA"
echo "-------------------------------------------------"

# 4. ResNet-18 + Bernoulli RAMA + AT (PGD)
echo "\nRunning: 4. ResNet-18 + Bernoulli RAMA + AT (PGD)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $RESNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $RESNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 4. ResNet-18 + Bernoulli RAMA + AT (PGD)"
echo "-------------------------------------------------"

# ==============================================================================
# ResNet-18 Gaussian RAMA Experiments for Tiny ImageNet
# ==============================================================================
PYTHON_SCRIPT_GAUSSIAN="Tiny_Imagenet/Resnet_multi_vector_rama_gaussian.py"
RESNET_RAMA_GAUSSIAN_ARGS="--use-rama --use-normalization --lambda-value 0.2 --activation relu --sqrt-dim False"

echo "\n\n-------------------------------------------------"
echo "Starting ResNet-18 (Gaussian RAMA) Experiments for Tiny ImageNet"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 5. ResNet-18 Baseline (Gaussian script, no RAMA) - Note: This is effectively the same as #1 but uses the Gaussian script
echo "\nRunning: 5. ResNet-18 Baseline (Gaussian Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 5. ResNet-18 Baseline (Gaussian Script)"
echo "-------------------------------------------------"

# 6. ResNet-18 + AT (PGD) (Gaussian script, no RAMA)
echo "\nRunning: 6. ResNet-18 + AT (PGD) (Gaussian Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 6. ResNet-18 + AT (PGD) (Gaussian Script)"
echo "-------------------------------------------------"

# 7. ResNet-18 + Gaussian RAMA
echo "\nRunning: 7. ResNet-18 + Gaussian RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $RESNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $RESNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 7. ResNet-18 + Gaussian RAMA"
echo "-------------------------------------------------"

# 8. ResNet-18 + Gaussian RAMA + AT (PGD)
echo "\nRunning: 8. ResNet-18 + Gaussian RAMA + AT (PGD)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $RESNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $RESNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 8. ResNet-18 + Gaussian RAMA + AT (PGD)"
echo "-------------------------------------------------"

echo "\nAll ResNet-18 Tiny ImageNet experiments finished." 