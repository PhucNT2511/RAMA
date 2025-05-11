from autoattack import AutoAttack
from CIFAR10_models import *
from utils import *
import argparse
import sys
import os
import logging
sys.path.insert(0, '..')
logger = logging.getLogger(__name__)


def parse_rama_positions(positions_str):
    """
    Parse comma-separated positions string into a dictionary for RAMA layer positions.
    
    Args:
        positions_str (str): Comma-separated string of positions (e.g. "layer1,layer2,final")
        
    Returns:
        dict: Dictionary mapping position names to boolean values
    """
    valid_positions = ['layer1', 'layer2', 'layer3', 'layer4', 'final']
    positions_dict = {pos: False for pos in valid_positions}
    
    if positions_str:
        selected_positions = [pos.strip() for pos in positions_str.split(',')]
        for pos in selected_positions:
            if pos in valid_positions:
                positions_dict[pos] = True
            else:
                logging.warning(f"Invalid RAMA position: {pos}. Skipping.")
    
    return positions_dict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='./model_test.pt')
    parser.add_argument('--model', default='ResNet18', type=str, help='model name')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--out_dir', type=str, default='./data')

    # Bernoulli RAMA configuration
    parser.add_argument('--use-rama', action='store_true', help='whether to use RAMA layers')
    parser.add_argument('--p-value', default=0.7, type=float, help='Bernoulli probability parameter (p-value)')
    parser.add_argument('--lambda-value', default=1.0, type=float, help='Lambda_value for RAMA')
    parser.add_argument('--sqrt-dim', action='store_true', help='Whether to divide by sqrt(d)')
    parser.add_argument('--bernoulli-values', default='0_1', choices=['0_1', '-1_1'],
                        type=str, help='values for Bernoulli distribution (0/1 or -1/1)')
    parser.add_argument('--use-normalization', action='store_true', help='use layer normalization in RAMA layers')
    parser.add_argument('--activation', default='silu', choices=['relu', 'leaky_relu', 'tanh', 'sigmoid', 'silu'],
                        help='activation function for RAMA layers')

    # RAMA position configuration
    parser.add_argument('--rama-positions', default='final',
                        type=str, help='comma-separated list of positions to apply RAMA (options: layer1,layer2,layer3,layer4,final)')

    parser.add_argument('--factor', default=0.7, type=float)
    parser.add_argument('--length', type=int, default=4, help='length of the holes')
    parser.add_argument('--n_holes', type=int, default=1, help='number of holes to cut out from image')
    parser.add_argument('--c_num', default=0.125, type=float)
    parser.add_argument('--EMA_value', default=0.82, type=float)

    arguments = parser.parse_args()
    return arguments


args = get_args()

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
logfile1 = os.path.join(args.out_dir, 'log_file1.txt')
logfile2 = os.path.join(args.out_dir, 'log_file2.txt')
if os.path.exists(logfile1):
    os.remove(logfile1)
if os.path.exists(logfile2):
    os.remove(logfile2)

# Configure RAMA positions
rama_positions = parse_rama_positions(args.rama_positions)
    
# Bernoulli RAMA configuration
rama_config = {
    "p_value": args.p_value,
    "values": args.bernoulli_values,
    "activation": args.activation,
    "use_normalization": args.use_normalization,
    "lambda_value": args.lambda_value,
    "sqrt_dim": args.sqrt_dim,
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.model == "VGG":
    target_model = VGG('VGG19')
elif args.model == "ResNet18":
    target_model = ResNet18(
        use_rama=args.use_rama,
        rama_config=rama_config,
        rama_positions=rama_positions,
        rama_type='bernoulli'
    )
elif args.model == "PreActResNest18":
    target_model = PreActResNet18(
        use_rama=args.use_rama,
        rama_config=rama_config,
        rama_positions=rama_positions,
        rama_type='bernoulli'
    )
elif args.model == "WideResNet":
    target_model = WideResNet(
        use_rama=args.use_rama,
        rama_config=rama_config,
        rama_positions=rama_positions,
        rama_type='bernoulli'
    )

target_model = target_model.to(device)
checkpoint = torch.load(args.model_path, weights_only=True)
from collections import OrderedDict
try:
    target_model.load_state_dict(checkpoint)
except:
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    target_model.load_state_dict(new_state_dict, False)



target_model.eval()
train_loader, test_loader = get_loaders(args.data_dir, args.batch_size)

# result=membership_inference_attack(target_model, train_loader, test_loader)
# print(result)
epsilon=args.epsilon
epsilon=float(epsilon)/255.
print(epsilon)
AT_fgsm_loss, AT_fgsm_acc=evaluate_fgsm(test_loader, target_model, 1)
AT_pgd_loss_10, AT_pgd_acc_10 = evaluate_pgd(test_loader, target_model, 10, 1, epsilon / std)
AT_pgd_loss_20, AT_pgd_acc_20 = evaluate_pgd(test_loader, target_model, 20, 1, epsilon / std)
AT_pgd_loss_50, AT_pgd_acc_50 = evaluate_pgd(test_loader, target_model, 50, 1, epsilon / std)
AT_CW_loss_20, AT_pgd_cw_acc_20 = evaluate_pgd_cw(test_loader, target_model, 20, 1)
AT_models_test_loss, AT_models_test_acc = evaluate_standard(test_loader, target_model)

print('AT_models_test_acc:', AT_models_test_acc)
print('AT_fgsm_acc:', AT_fgsm_acc)
print('AT_pgd_acc_10:', AT_pgd_acc_10)
print('AT_pgd_acc_20:', AT_pgd_acc_20)
print('AT_pgd_acc_50:', AT_pgd_acc_50)
print('AT_pgd_cw_acc_20:', AT_pgd_cw_acc_20)






adversary1 = AutoAttack(target_model, norm=args.norm, eps=epsilon, version='standard',log_path=logfile1)

#adversary2 = AutoAttack(target_model, norm=args.norm, eps=epsilon, version='standard',log_path=logfile2)
l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0)

adv_complete = adversary1.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.batch_size)
# adv_complete1 = adversary2.run_standard_evaluation_individual(x_test[:args.n_ex], y_test[:args.n_ex],
#                 bs=args.batch_size)