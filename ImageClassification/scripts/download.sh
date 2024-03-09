# wget -nc https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# mkdir datasets/CIFAR10
# tar -zxvf cifar-10-python.tar.gz --directory datasets/CIFAR10 -k 
# rm cifar-10-python.tar.gz

wget -nc https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
mkdir datasets/fgvc
tar -zxvf fgvc-aircraft-2013b.tar.gz --directory datasets/fgvc -k 
rm fgvc-aircraft-2013b.tar.gz