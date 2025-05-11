import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class DataManager:
    """
    Manager for CIFAR-10 dataset preparation and loading.
    
    Args:
        data_dir (str): Directory to store/load dataset.
        batch_size (int): Batch size for data loaders.
        num_workers (int): Number of workers for data loading. Default: 2.
    """
    def __init__(self, data_dir, batch_size, num_workers=2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
    def get_loaders(self):
        """
        Get data loaders for training and testing.
        
        Returns:
            tuple: (train_loader, test_loader)
        """
        # Training dataset
        trainset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=self.transform_train
        )
        trainloader = DataLoader(
            trainset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

        # Testing dataset
        testset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=self.transform_test
        )
        testloader = DataLoader(
            testset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        return trainloader, testloader