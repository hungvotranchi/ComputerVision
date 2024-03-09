from torchvision import datasets
from src.preprocess import transform_CIFAR10
from torchvision.transforms import transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def load_CIFAR10(transform):
    data_path = r"..\datasets\CIFAR10"
    cifar10 = datasets.CIFAR10(root=data_path, train = True, \
                               download= False, transform= transform)
    cifar10_val = datasets.CIFAR10(root=data_path, train = False, \
                               download= False, transform= transform)
    return cifar10, cifar10_val