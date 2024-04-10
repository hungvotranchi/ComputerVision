# wget -nc https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
# mkdir datasets/CIFAR10
# tar -zxvf cifar-10-python.tar.gz --directory datasets/CIFAR10 -k 
# rm cifar-10-python.tar.gz

wget -nc http://images.cocodataset.org/zips/train2017.zip
wget -nc http://images.cocodataset.org/zips/val2017.zip
wget -nc http://images.cocodataset.org/zips/test2017.zip
wget -nc http://images.cocodataset.org/zips/unlabeled2017.zip

mkdir datasets/COCO

unzip train2017.zip -d datasets/COCO
unzip val2017.zip -d datasets/COCO
unzip test2017.zip -d datasets/COCO
unzip unlabeled2017.zip -d datasets/COCO

rm train2017.zip
rm val2017.zip
rm test2017.zip
rm unlabeled2017.zip 

wget -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -nc http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget -nc http://images.cocodataset.org/annotations/image_info_test2017.zip
wget -nc http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

unzip annotations_trainval2017.zip -d datasets/COCO
unzip stuff_annotations_trainval2017.zip -d datasets/COCO
unzip image_info_test2017.zip -d datasets/COCO
unzip image_info_unlabeled2017.zip -d datasets/COCO

rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip
rm image_info_test2017.zip
rm image_info_unlabeled2017.zip