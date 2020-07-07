# TF2_TinyYOLOv4
TinyYOLOv4 Tensorflow2-keras version  
Tiny YOLOv4的tf.keras版本，如果需要加入DropBlock可以參考[TF2_YOLOv4](https://github.com/qwerasdf887/TF2_YOLOv4/blob/master/model/models.py)  
Tiny v4，使用與v3相同的方式：  
Conv2D -> BN -> Leakly，當作一個Conv單元

## 環境 environment 

1. Tensorflow 2.2
2. tensorflow_addons (moving average opt)
3. Python 3.5~3.7
4. OpenCV 3~4

## Tiny Weights

ALexeyAB : [download weight](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)

## Predict Img:

```bashrc
python predict.py
```

## Result

<p align="center">
    <img width="100%" src="https://github.com/qwerasdf887/TF2_TinyYOLOv4/blob/master/predict.jpg?raw=true" style="max-width:100%;">
    </a>
</p>

## Training

需修改train.py，沒有寫得很完整，有問題可以發issue。
使用[labelImg](https://github.com/tzutalin/labelImg) 生成的xml檔，並且放入標籤位置即可。