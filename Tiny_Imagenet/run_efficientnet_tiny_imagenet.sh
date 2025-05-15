#!/bin/bash

# --- Configuration for Adversarial Evaluation and Training ---
EVAL_FGSM=true
EVAL_PGD=true
AT_MODE_PGD="--at-attack pgd --at-epsilon 0.03 --at-alpha 0.01 --at-iter 7" # AT PGD specific params

# --- Base command arguments (common to all EfficientNet-B2 runs on Tiny ImageNet) ---
BASE_ARGS="--epochs 100 --lr 0.01 --batch-size 64 --seed 42 --checkpoint-dir ./checkpoints_tiny_imagenet/EfficientNetB2" # Adjusted batch-size for potential memory constraints
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
# EfficientNet-B2 Bernoulli RAMA Experiments for Tiny ImageNet
# ==============================================================================
PYTHON_SCRIPT_BERNOULLI="Tiny_Imagenet/EffficientNet_multi_vector_rama_bernoulli.py"
EFFICIENTNET_RAMA_BERNOULLI_ARGS="--use-rama --use-normalization --p-value 0.7 --bernoulli-values=0_1 --activation relu --sqrt-dim False"

echo "-------------------------------------------------"
echo "Starting EfficientNet-B2 (Bernoulli RAMA) Experiments for Tiny ImageNet"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 1. EfficientNet-B2 Baseline (Bernoulli script, no RAMA)
echo "\nRunning: 1. EfficientNet-B2 Baseline (Bernoulli Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 1. EfficientNet-B2 Baseline (Bernoulli Script)"
echo "-------------------------------------------------"

# 2. EfficientNet-B2 + AT (PGD) (Bernoulli script, no RAMA)
echo "\nRunning: 2. EfficientNet-B2 + AT (PGD) (Bernoulli Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 2. EfficientNet-B2 + AT (PGD) (Bernoulli Script)"
echo "-------------------------------------------------"

# 3. EfficientNet-B2 + Bernoulli RAMA
echo "\nRunning: 3. EfficientNet-B2 + Bernoulli RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $EFFICIENTNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $EFFICIENTNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 3. EfficientNet-B2 + Bernoulli RAMA"
echo "-------------------------------------------------"

# 4. EfficientNet-B2 + Bernoulli RAMA + AT (PGD)
echo "\nRunning: 4. EfficientNet-B2 + Bernoulli RAMA + AT (PGD)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $EFFICIENTNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_BERNOULLI $BASE_ARGS $DATASET_ARGS $EFFICIENTNET_RAMA_BERNOULLI_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 4. EfficientNet-B2 + Bernoulli RAMA + AT (PGD)"
echo "-------------------------------------------------"

# ==============================================================================
# EfficientNet-B2 Gaussian RAMA Experiments for Tiny ImageNet
# ==============================================================================
PYTHON_SCRIPT_GAUSSIAN="Tiny_Imagenet/EffficientNet_multi_vector_rama_gaussian.py"
EFFICIENTNET_RAMA_GAUSSIAN_ARGS="--use-rama --use-normalization --lambda-value 0.05 --activation relu --sqrt-dim False --sigma-p-value 0.05" # sigma-p-value might be specific here

echo "\n\n-------------------------------------------------"
echo "Starting EfficientNet-B2 (Gaussian RAMA) Experiments for Tiny ImageNet"
echo "Evaluations enabled: FGSM=$EVAL_FGSM, PGD=$EVAL_PGD"
echo "-------------------------------------------------"

# 5. EfficientNet-B2 Baseline (Gaussian script, no RAMA)
echo "\nRunning: 5. EfficientNet-B2 Baseline (Gaussian Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 5. EfficientNet-B2 Baseline (Gaussian Script)"
echo "-------------------------------------------------"

# 6. EfficientNet-B2 + AT (PGD) (Gaussian script, no RAMA)
echo "\nRunning: 6. EfficientNet-B2 + AT (PGD) (Gaussian Script)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 6. EfficientNet-B2 + AT (PGD) (Gaussian Script)"
echo "-------------------------------------------------"

# 7. EfficientNet-B2 + Gaussian RAMA
echo "\nRunning: 7. EfficientNet-B2 + Gaussian RAMA"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $EFFICIENTNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $EFFICIENTNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 7. EfficientNet-B2 + Gaussian RAMA"
echo "-------------------------------------------------"

# 8. EfficientNet-B2 + Gaussian RAMA + AT (PGD)
echo "\nRunning: 8. EfficientNet-B2 + Gaussian RAMA + AT (PGD)"
echo "Command: CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $EFFICIENTNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS"
CUDA_VISIBLE_DEVICES=0 python $PYTHON_SCRIPT_GAUSSIAN $BASE_ARGS $DATASET_ARGS $EFFICIENTNET_RAMA_GAUSSIAN_ARGS $ADVERSARIAL_TRAINING_FLAGS_PGD $ADVERSARIAL_EVAL_FLAGS
echo "Finished: 8. EfficientNet-B2 + Gaussian RAMA + AT (PGD)"
echo "-------------------------------------------------"

echo "\nAll EfficientNet-B2 Tiny ImageNet experiments finished." 