import os
import logging
import torch
from tqdm import tqdm
from bayes_opt import BayesianOptimization, acquisition

from .metrics import calculate_feature_metrics
from .attacks import fgsm_attack, pgd_attack

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for model training and evaluation with feature quality metrics.
    
    Args:
        model: Model to train
        trainloader: Training data loader
        testloader: Testing data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use for training
        checkpoint_dir: Directory to save checkpoints
        bayes_opt_config: Configuration for Bayesian optimization
        use_rama: Whether the model uses RAMA layers
        use_hyperparameter_optimization: Whether to use Bayesian optimization
        scheduler: Learning rate scheduler
        neptune_run: Neptune.ai run instance
        writer: TensorBoard SummaryWriter
        eval_fgsm (bool): Whether to evaluate with FGSM attack.
        eval_pgd (bool): Whether to evaluate with PGD attack.
        adv_epsilon (float): Epsilon for adversarial attacks.
        pgd_alpha (float): Alpha for PGD attack.
        pgd_iter (int): Number of iterations for PGD attack.
        adversarial_training_attack (str): Type of attack for adversarial training (None, 'fgsm', 'pgd').
    """
    def __init__(self, model, trainloader, testloader, criterion, optimizer, 
                 device, checkpoint_dir, bayes_opt_config=None, use_rama=False,
                 use_hyperparameter_optimization=False, scheduler=None,
                 neptune_run=None, writer=None,
                 eval_fgsm=False, eval_pgd=False, adv_epsilon=8/255, 
                 pgd_alpha=2/255, pgd_iter=10,
                 adversarial_training_attack=None):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.best_acc = 0
        self.neptune_run = neptune_run
        self.writer = writer
        self.use_rama = use_rama
        self.use_hyperparameter_optimization = use_hyperparameter_optimization
        self.best_p = None
        self.scheduler = scheduler
        
        # Store adversarial evaluation parameters
        self.eval_fgsm = eval_fgsm
        self.eval_pgd = eval_pgd
        self.adv_epsilon = adv_epsilon
        self.pgd_alpha = pgd_alpha
        self.pgd_iter = pgd_iter
        self.adversarial_training_attack = adversarial_training_attack

        if self.use_rama and self.use_hyperparameter_optimization:
            # Default Bayesian optimization configuration
            self.bayes_opt_config = bayes_opt_config

            # Initialize Bayesian optimizer with correct bounds
            if self.bayes_opt_config["acq"] == "ei":
                acq = acquisition.ExpectedImprovement(xi=self.bayes_opt_config["xi"])
            elif self.bayes_opt_config["acq"] == "poi":
                acq = acquisition.ProbabilityOfImprovement(xi=self.bayes_opt_config["xi"])
            elif self.bayes_opt_config["acq"] == "ucb":
                acq = acquisition.UpperConfidenceBound(kappa=self.bayes_opt_config["kappa"])
            else:
                raise ValueError("Invalid acquisition function specified.")
                
            self.bayesian_optimizer = BayesianOptimization(
                f=self.evaluate_p,
                acquisition_function=acq,
                pbounds={"p_value": (self.bayes_opt_config["p_min"], self.bayes_opt_config["p_max"])},
                random_state=42,
                verbose=2
            )

    def optimize_p(self, n_warmup=None, n_iter=None):
        """
        Run Bayesian optimization to find the best p value.
        
        Args:
            n_warmup: Number of random points to evaluate
            n_iter: Number of iterations
            
        Returns:
            tuple: (best_p, best_score, optimization_results)
        """
        if n_warmup is None:
            n_warmup = self.bayes_opt_config["init_points"]
        if n_iter is None:
            n_iter = self.bayes_opt_config["n_iter"]
            
        logger.info(f"Running Bayesian optimization with {n_warmup} initialization points and {n_iter} iterations...")
        
        # Run optimization
        self.bayesian_optimizer.maximize(
            init_points=n_warmup,
            n_iter=n_iter
        )
        
        best_p = self.bayesian_optimizer.max["params"]["p_value"]
        best_score = self.bayesian_optimizer.max["target"]
        return best_p, best_score, self.bayesian_optimizer.res

    def load_optimizer_state(self, path):
        """
        Load Bayesian optimizer state from a file.
        
        Args:
            path: Path to the state file
        """
        if os.path.exists(path):
            logger.info(f"Loading BayesOpt state from {path}")
            self.bayesian_optimizer.load_state(path)
            logger.info(f"Loaded max value: {self.bayesian_optimizer.max}")

    def train_one_epoch(self, p_value=None):
        """
        Train the model for one epoch.
        
        Args:
            p_value: Value for RAMA p parameter
            
        Returns:
            tuple: (train_loss, train_accuracy)
        """
        self.model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.trainloader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # If adversarial training is enabled, generate adversarial examples
            current_inputs = inputs
            if self.adversarial_training_attack == 'fgsm':
                # Ensure model is in eval mode for attack generation, then back to train
                self.model.eval()
                current_inputs = fgsm_attack(self.model, inputs, targets, self.adv_epsilon, self.device)
                self.model.train()
            elif self.adversarial_training_attack == 'pgd':
                # Ensure model is in eval mode for attack generation, then back to train
                self.model.eval()
                current_inputs = pgd_attack(self.model, inputs, targets, 
                                            self.adv_epsilon, self.pgd_alpha, self.pgd_iter, self.device)
                self.model.train()
            
            self.optimizer.zero_grad()
            # Use current_inputs (which may be adversarial) for the forward pass
            outputs = self.model.forward(current_inputs, p_value=p_value)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar.set_postfix({
                "loss": train_loss / (batch_idx + 1),
                "acc": 100. * correct / total
            })
        return train_loss / len(self.trainloader), 100. * correct / total

    def evaluate(self, p_value=None):
        """
        Basic evaluation of the model on the test set.
        
        Args:
            p_value: Value for RAMA p parameter
            
        Returns:
            tuple: (test_loss, test_accuracy)
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            pbar = tqdm(self.testloader, desc="Testing (Clean)")
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model.forward(inputs, p_value=p_value)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': test_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        return test_loss / len(self.testloader), 100. * correct / total

    def evaluate_fgsm(self, p_value=None):
        """
        Evaluate the model on FGSM adversarial examples.
        
        Args:
            p_value: Value for RAMA p parameter (if model uses it).

        Returns:
            tuple: (fgsm_loss, fgsm_accuracy)
        """
        self.model.eval()
        fgsm_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.testloader, desc="Testing (FGSM)")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Generate FGSM adversarial examples
            # Note: FGSM attack enables gradients, so no torch.no_grad() here for attack generation
            adv_inputs = fgsm_attack(self.model, inputs, targets, self.adv_epsilon, self.device)
            
            with torch.no_grad(): # Disable gradients for evaluation pass
                outputs = self.model.forward(adv_inputs, p_value=p_value)
                loss = self.criterion(outputs, targets)
                
                fgsm_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': fgsm_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        return fgsm_loss / len(self.testloader), 100. * correct / total

    def evaluate_pgd(self, p_value=None):
        """
        Evaluate the model on PGD adversarial examples.

        Args:
            p_value: Value for RAMA p parameter (if model uses it).
            
        Returns:
            tuple: (pgd_loss, pgd_accuracy)
        """
        self.model.eval()
        pgd_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(self.testloader, desc="Testing (PGD)")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Generate PGD adversarial examples
            # Note: PGD attack enables gradients, so no torch.no_grad() here for attack generation
            adv_inputs = pgd_attack(self.model, inputs, targets, 
                                    self.adv_epsilon, self.pgd_alpha, self.pgd_iter, self.device)

            with torch.no_grad(): # Disable gradients for evaluation pass
                outputs = self.model.forward(adv_inputs, p_value=p_value)
                loss = self.criterion(outputs, targets)
                
                pgd_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'loss': pgd_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        return pgd_loss / len(self.testloader), 100. * correct / total

    def evaluate_p(self, p_value):
        """
        Evaluation function for Bayesian optimization.
        Returns a single scalar value (accuracy).
        
        Args:
            p_value: Value for RAMA p parameter
            
        Returns:
            float: Test accuracy
        """
        _, test_acc = self.evaluate(p_value)
        return test_acc  # Return only accuracy for optimization

    def evaluate_with_metrics(self, p_value=None):
        """
        Evaluate the model with additional metrics to understand RAMA impact.
        
        Args:
            p_value: Value for RAMA p parameter
            
        Returns:
            dict: Evaluation metrics including feature quality
        """
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        # Initialize containers for feature analysis
        features_original = []
        features_after_rama = []
        class_labels = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Use the forward pass that returns features
                if hasattr(self.model, "forward_with_features"):
                    outputs, before_features, after_features = self.model.forward_with_features(inputs, p_value)
                    if before_features is not None and after_features is not None:
                        features_original.append(before_features.cpu())
                        features_after_rama.append(after_features.cpu())
                        class_labels.append(targets.cpu())
                else:
                    outputs = self.model.forward(inputs, p_value)
                
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate standard metrics
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(self.testloader)
        
        # Analyze features if we captured them
        feature_metrics = {}
        if features_original and features_after_rama:
            # Concatenate batches
            features_original = torch.cat(features_original, dim=0)
            features_after_rama = torch.cat(features_after_rama, dim=0)
            class_labels = torch.cat(class_labels, dim=0)
            
            # Calculate feature separability before and after RAMA
            feature_metrics = calculate_feature_metrics(
                features_original, features_after_rama, class_labels)
            
            # Log feature metrics
            if self.neptune_run:
                for key, value in feature_metrics.items():
                    self.neptune_run[f"Feature/{key}"].append(value)
            
            if self.writer:
                for key, value in feature_metrics.items():
                    self.writer.add_scalar(f"Feature/{key}", value)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'feature_metrics': feature_metrics
        }

    def save_checkpoint(self, state, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            state: State to save
            is_best: Whether this is the best model so far
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        logger.info(f"Saving checkpoint to {checkpoint_path}")
        torch.save(state, checkpoint_path)
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            logger.info(f"Saving best model to {best_path}")
            torch.save(state, best_path)
    
    def train(self, epochs, start_epoch=0):
        """
        Train the model for multiple epochs with feature quality evaluation.
        
        Args:
            epochs: Number of epochs to train
            start_epoch: Starting epoch
            
        Returns:
            float: Best test accuracy
        """
        # First, run initial optimization to find a good p
        if self.use_rama and self.use_hyperparameter_optimization:
            if start_epoch == 0:
                logger.info("Running initial Bayesian optimization...")
                self.best_p, _, _ = self.optimize_p()
            else:
                # Try to load previous optimization state
                bayes_opt_path = os.path.join(self.checkpoint_dir, 'bayes_opt_state.json')
                self.load_optimizer_state(bayes_opt_path)
                if self.bayesian_optimizer.max:
                    self.best_p = self.bayesian_optimizer.max["params"]["p_value"]
                    logger.info(f"Loaded best p from previous run: {self.best_p:.6f}")
                else:
                    logger.info("No previous optimization state found, running initial optimization...")
                    self.best_p, _, _ = self.optimize_p()

        for epoch in range(start_epoch, epochs):
            logger.info(f"\nEpoch: {epoch+1}/{epochs}")

            # Update RAMA masks if the model has a method to update them
            if self.use_rama and self.best_p is not None and hasattr(self.model, "update_rama_masks"):
                self.model.update_rama_masks(self.best_p)

            # Train with best p
            train_loss, train_acc = self.train_one_epoch(p_value=self.best_p)
            
            # Basic evaluation (Clean Accuracy)
            test_loss, test_acc = self.evaluate(p_value=self.best_p)
            
            # Adversarial Evaluation if enabled
            fgsm_acc, pgd_acc = -1.0, -1.0 # Default if not evaluated
            fgsm_loss, pgd_loss = -1.0, -1.0
            if test_acc > self.best_acc and (epoch % 15 == 0 or epoch == epochs - 1):
                if self.eval_fgsm:
                    fgsm_loss, fgsm_acc = self.evaluate_fgsm(p_value=self.best_p)

                if self.eval_pgd:
                    pgd_loss, pgd_acc = self.evaluate_pgd(p_value=self.best_p)

            # Detailed evaluation with feature metrics (once every 5 epochs to save time)
            if epoch % 5 == 0 or epoch == epochs - 1:
                metrics = self.evaluate_with_metrics(p_value=self.best_p)
                if 'feature_metrics' in metrics and metrics['feature_metrics']:
                    feature_metrics = metrics['feature_metrics']
                    logger.info(f"Feature metrics at epoch {epoch+1}:")
                    logger.info(f"  Fisher ratio before RAMA: {feature_metrics['fisher_ratio_before']:.4f}")
                    logger.info(f"  Fisher ratio after RAMA: {feature_metrics['fisher_ratio_after']:.4f}")
                    logger.info(f"  Fisher improvement: {feature_metrics['fisher_improvement']:.4f}x")

            # Step the scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(test_acc)  # For plateau scheduler, use accuracy
                else:
                    self.scheduler.step()  # For other schedulers, step automatically
                
                # Log current learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Current learning rate: {current_lr:.6f}")
                if self.neptune_run:
                    self.neptune_run["learning_rate"].append(current_lr)
                if self.writer:
                    self.writer.add_scalar("learning_rate", current_lr, epoch)

            # Perform Bayesian optimization periodically
            if (self.use_rama and self.use_hyperparameter_optimization and
                    epoch % self.bayes_opt_config["optimize_every"] == 0 and epoch > 0):
                logger.info(f"Running Bayesian optimization at epoch {epoch+1}...")
                # Use fewer iterations for subsequent optimizations
                p_value, bayesian_score, results = self.optimize_p(
                    n_warmup=max(2, self.bayes_opt_config["init_points"] // 2), 
                    n_iter=max(5, self.bayes_opt_config["n_iter"] // 2)
                )
                if bayesian_score >= test_acc:
                    self.best_p = p_value
                    logger.info(f"Updated best p: {self.best_p:.4f} with accuracy: {bayesian_score:.2f}%")

                    # Update the search bounds based on the best p found
                    p_min_distance = abs(p_value - self.bayes_opt_config["p_min"])
                    p_max_distance = abs(p_value - self.bayes_opt_config["p_max"])
                    
                    # If closer to min bound, expand upper bound
                    if p_min_distance < p_max_distance:
                        new_max = min(self.best_p * 1.5, self.bayes_opt_config["p_max"] * 2)
                        self.bayesian_optimizer.set_bounds(
                            new_bounds={"p_value": (self.bayes_opt_config["p_min"], new_max)}
                        )
                    # If closer to max bound, expand lower bound
                    else:
                        new_min = max(self.best_p * 0.5, self.bayes_opt_config["p_min"] * 0.5)
                        self.bayesian_optimizer.set_bounds(
                            new_bounds={"p_value": (new_min, self.bayes_opt_config["p_max"])}
                        )

            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Test Loss: {test_loss:.4f} | Test Acc (Clean): {test_acc:.2f}%")
            if self.eval_fgsm:
                logger.info(f"FGSM Loss: {fgsm_loss:.4f} | FGSM Acc: {fgsm_acc:.2f}%")
            if self.eval_pgd:
                logger.info(f"PGD Loss: {pgd_loss:.4f} | PGD Acc: {pgd_acc:.2f}%")

            if self.use_rama and self.use_hyperparameter_optimization:
                logger.info(f"Current p value: {self.best_p:.6f}")

            # Log to Neptune if available
            if self.neptune_run:
                self.neptune_run["Train/Loss"].append(train_loss)
                self.neptune_run["Train/Accuracy"].append(train_acc)
                self.neptune_run["Test/Loss"].append(test_loss)
                self.neptune_run["Test/Accuracy"].append(test_acc) # Clean accuracy
                if self.eval_fgsm:
                    self.neptune_run["Test/FGSM_Loss"].append(fgsm_loss)
                    self.neptune_run["Test/FGSM_Accuracy"].append(fgsm_acc)
                if self.eval_pgd:
                    self.neptune_run["Test/PGD_Loss"].append(pgd_loss)
                    self.neptune_run["Test/PGD_Accuracy"].append(pgd_acc)
                if self.use_rama and self.use_hyperparameter_optimization:
                    self.neptune_run["RAMA_P"].append(self.best_p)

            # Log to TensorBoard if available
            if self.writer:
                self.writer.add_scalar("Train/Loss", train_loss, epoch)
                self.writer.add_scalar("Train/Accuracy", train_acc, epoch)
                self.writer.add_scalar("Test/Loss", test_loss, epoch)
                self.writer.add_scalar("Test/Accuracy", test_acc, epoch) # Clean accuracy
                if self.eval_fgsm:
                    self.writer.add_scalar("Test/FGSM_Loss", fgsm_loss, epoch)
                    self.writer.add_scalar("Test/FGSM_Accuracy", fgsm_acc, epoch)
                if self.eval_pgd:
                    self.writer.add_scalar("Test/PGD_Loss", pgd_loss, epoch)
                    self.writer.add_scalar("Test/PGD_Accuracy", pgd_acc, epoch)
                if self.use_rama and self.use_hyperparameter_optimization:
                    self.writer.add_scalar("RAMA_P", self.best_p, epoch)

            # Save checkpoint
            is_best = test_acc > self.best_acc
            if is_best:
                self.best_acc = test_acc
                checkpoint_state = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_acc": self.best_acc,
                    "best_p": self.best_p,
                }
                # Add scheduler state if it exists
                if self.scheduler:
                    checkpoint_state["scheduler_state_dict"] = self.scheduler.state_dict()
                self.save_checkpoint(checkpoint_state, is_best)
                
                # Save Bayesian optimization state
                if self.use_rama and self.use_hyperparameter_optimization:
                    bayes_opt_path = os.path.join(self.checkpoint_dir, 'bayes_opt_state.json')
                    self.bayesian_optimizer.save_state(bayes_opt_path)
                
        logger.info(f"Best test accuracy: {self.best_acc:.2f}%")
        return self.best_acc
