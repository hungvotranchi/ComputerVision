wget -nc https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
mkdir datasets/CIFAR10
tar -zxvf cifar-10-python.tar.gz --directory datasets/CIFAR10 -k 
rm cifar-10-python.tar.gz