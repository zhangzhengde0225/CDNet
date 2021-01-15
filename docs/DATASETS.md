# DATASETS

##### Download datasets for CDNet

**Trainset**
+ from jbox: [https://jbox.sjtu.edu.cn/l/Z0i6nQ](https://jbox.sjtu.edu.cn/l/Z0i6nQ)
+ or use wget:
```python
wget -c -O train_data_v5_format_3080.tar https://jbox.sjtu.edu.cn:10081/v2/delivery/data/de2ac2610893499686d095f174aa6ffd/?token=
```
**notes**

Please remove proxy if download failed.

Unzip:
```python
tar -xvf train_data_v5_format_3080.tar
cd train_data_v5_format_3080
rm labels/*.cache  # if you are the first time to use this datasets.

```
The trainsets includes training and verification set, and the file structure uses the format of YOLOv5:
```python
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
Each .txt file contains annotations in the form of CLS XC YC W H in each line.


**Testsets**
+ from jbox: [https://jbox.sjtu.edu.cn/l/eHE7vD](https://jbox.sjtu.edu.cn/l/eHE7vD)
+ or use wget:
```python
wget -c -O testsets_1770.tar https://jbox.sjtu.edu.cn:10081/v2/delivery/data/a526b168356949068e6a13e8c254a71b/?token=
```

Unzip:
```python
tar -xvf testsets_1770.tar
cd testsets_1770
```



