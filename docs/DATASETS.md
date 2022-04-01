# DATASETS

The crosswalk detection datasets contains 3080 images for train and 1770 images for test.

## Trainset
+ Download from [Baidu Pan](https://pan.baidu.com/s/1jAsQ_kbU7cX0AqK4wDm6LA), passwd: **hul6**
+ Download from [Google Drive](https://drive.google.com/file/d/1RIIpdrAUXZRuoOgHMIr-BSsJxSjqiKC8/view?usp=sharing).
+ Download from [SJTU Jbox](https://jbox.sjtu.edu.cn/l/Z0i6nQ).
+ or use wget:
```
wget -c -O train_data_v5_format_3080.tar https://jbox.sjtu.edu.cn:10081/v2/delivery/data/de2ac2610893499686d095f174aa6ffd/?token=
```
**Notes**

Please remove proxy if download failed.

Unzip:
```
tar -xvf train_data_v5_format_3080.tar
cd train_data_v5_format_3080
rm labels/*.cache  # if you are the first time to use this datasets.
```
The trainsets contains 3080 images, includes training and verification set, and the file structure format is YOLOv5 likes:
```
|--train_data_v5_format
--|--images  # the .jpg images
   --|--train  # trainning images
      -- xxx.jpg
      -- ...
   --|--test  # validation images
--|--labels  # corresponding .txt labels
   --|--train  # trainning labels
      -- xxx.txt
      -- ...
   --|--test  # validation labels
```
Each .txt file contains annotations in the format of CLS XC YC W H in each line. 

CLS(Classes): crosswalk, guide_arrows

XC YC W H in terms of percentage.


## Testsets
+ Download from [Baidu Pan](https://pan.baidu.com/s/1-AWw2AjE8zTP-iVjOoifOg), passwd: **vj6b**
+ Download from [Google Drive](https://drive.google.com/file/d/1DBVt81LehEyiTxuUn-BUGBoh3gXcwrEE/view?usp=sharing).
+ Download from [SJTU Jbox](https://jbox.sjtu.edu.cn/l/eHE7vD).
+ or use wget:
```
wget -c -O testsets_1770.tar https://jbox.sjtu.edu.cn:10081/v2/delivery/data/a526b168356949068e6a13e8c254a71b/?token=
```

Unzip:
```
tar -xvf testsets_1770.tar
cd testsets_1770
```
The testsets contains 1770 images.

Currently we do not provide the testsets for vehicle crossing behavior analysis.



