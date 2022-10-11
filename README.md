# YOLOX train your data

you need generate data.txt like follow format **(per line-> one image)**.

## **prepare one data.txt like this:**

*img_path1 x1,y1,x2,y2,class_id x1,y1,x2,y2,class_id2*  
*img_path2 x1,y1,x2,y2,class_id*  
*img_path3 ..........*  

### **note:**

x1,y1,x2,y2 is int type and it belong to 0-img_w ,0-img_h, not 0~1 !!!  
img_path is abs path  
must be careful the sign " " and "," in data.txt  

**there was an example:**  
*/home/sal/images/000010.jpg 0,190,466,516,1*  
*/home/sal/images/000011.jpg 284,548,458,851,7 256,393,369,608,1*  

## **Train**

### i. step1 , before train, you need modify
**/content/YOLOX-train-your-data/yolox/exp/yolox_base.py**
```python
from yolox.exp.base_exp import BaseExp
```
> 1. num_classes
> 2. train_txt
> 3. val_txt

**/content/YOLOX-train-your-data/yolox/data/datasets/my_classes.py**
> modify class

**/content/YOLOX-train-your-data/train.py**
> line 16:#

**CUDA out of memory. Tried to allocate 100.00 MiB (GPU 0; 15.90 GiB total capacity; 14.75 GiB already allocated; 45.75 MiB free; 14.91 GiB reserved in total by PyTorch)**
> decrease --batch-size

### ii. step2 ,download weights from
https://github.com/Megvii-BaseDetection/YOLOX

### iii. step3 ,start

```python
python train.py -f yolox/exp/yolox_base.py -c yolox_nano.pth -d 1 --batch-size 8
```

### iii. step3 ,change train.py params by
https://github.com/Megvii-BaseDetection/YOLOX.git

```python
python train.py
```
