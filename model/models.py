import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
from tensorflow.keras.regularizers import l2
import numpy as np

'''
Tiny-YOLOv4使用的Conv與v3相同
Conv2D -> BN -> Leaky
比較小的Res Block與CSP Block
'''


def unit_conv(tensor, num_filters, k_size, strides, use_bn=True):
    #使用BN層，就不會使用conv的bias。
    #在YOLO中，padding與一般padding不同，只補一行一列
    padding = 'valid' if strides == 2 else 'same'
    if use_bn:
        #downsampling
        if strides == 2:
            tensor = tf.keras.layers.ZeroPadding2D(((1,0), (1,0)))(tensor)
        output = tf.keras.layers.Conv2D(filters=num_filters,
                                        kernel_size=k_size,
                                        strides=strides,
                                        padding=padding,
                                        kernel_regularizer=l2(1e-5),
                                        use_bias=False)(tensor)
        output = tf.keras.layers.BatchNormalization()(output)
        output = tf.keras.layers.LeakyReLU(0.1)(output)
        
    else:
        output = tf.keras.layers.Conv2D(filters=num_filters,
                                        kernel_size=k_size,
                                        padding=padding,
                                        kernel_regularizer=l2(1e-5))(tensor)
    return output

def Res_Block(tensor, num_filters):
    x_1 = unit_conv(tensor=tensor, num_filters=num_filters, k_size=3, strides=1)
    x_2 = unit_conv(tensor=x_1, num_filters=num_filters, k_size=3, strides=1)
    x = tf.keras.layers.Concatenate()([x_2, x_1])
    x = unit_conv(tensor=x, num_filters=num_filters*2, k_size=1, strides=1)
    return x

def CSP_Block(tensor, num_filters, first=False):
    '''
    輸入的tensor形成兩路地一路經過一個conv與第二路的Res Block合併
    若為第一次呼叫，使用Conv做下採樣，否則使用maxpooling
    '''
    if first:
        tensor = unit_conv(tensor=tensor, num_filters=num_filters, k_size=3, strides=2)
    else:
        tensor = tf.keras.layers.MaxPooling2D(2, 2, 'same')(tensor)

    #part 1
    part_1 = unit_conv(tensor=tensor, num_filters=num_filters, k_size=3, strides=1)

    #part 2
    sp = tf.split(part_1, num_or_size_splits=2, axis=-1)
    res_out = Res_Block(sp[1], num_filters=num_filters // 2)
    output = tf.keras.layers.Concatenate()([part_1, res_out])
    return output, res_out

'''
將yolo輸出座標(xmin, ymin, xmax, ymax)、confidence、類別機率
input shape: (batch size, grid, grid, 255)
'''
class yolo_head(tf.keras.layers.Layer):
    def __init__(self, anchors, num_cls, img_shape):
        super(yolo_head, self).__init__()
        self.anchors = anchors
        self.num_anc = len(anchors)
        self.num_cls = num_cls
        self.scale_x_y = 1.05
        #shape = (h, w)
        self.shape = img_shape[:2]
    
    def call(self, inputs):
        #限制inputs tensor為tf.float32 type
        x = tf.cast(inputs, tf.float32)
        #reshape layer to (batch size, gird, grid, anchors pre layer, (5+cls))
        x = tf.reshape(x, [-1, self.grid_shape[0], self.grid_shape[1], self.num_anc, (5+self.num_cls)])
        '''
        x_cen, y_cen = x[...,0:2]取sigmoid + cx, cy，再除以該feature map大小，獲得原圖的歸一化數據
        yolov4中引入"Eliminate grid sensitivity"，利用一個平移參數調整極端值問題
        '''
        box_xy = tf.sigmoid(x[...,0:2]) * self.scale_x_y - ((self.scale_x_y - 1) * 0.5) + self.grid
        #計算中心點於原始image的位置
        box_xy = (box_xy / tf.cast(self.grid_shape[::-1], tf.float32)) * tf.cast(self.shape[::-1], tf.float32)
        #w,h:取exp再乘以anchors得到實際寬高
        box_wh = tf.exp(x[...,2:4]) * self.anchors
        #計算(xmin, ymin, xmax, ymax)，並且限制範圍
        box_xmin, box_ymin = tf.split(box_xy - (box_wh / 2), num_or_size_splits=2, axis=-1)
        box_xmax, box_ymax = tf.split(box_xy + (box_wh / 2), num_or_size_splits=2, axis=-1)
        box_xmin = tf.clip_by_value(box_xmin, 0, self.shape[1])
        box_ymin = tf.clip_by_value(box_ymin, 0, self.shape[0])
        box_xmax = tf.clip_by_value(box_xmax, 0, self.shape[1])
        box_ymax = tf.clip_by_value(box_ymax, 0, self.shape[0])
        #形成(xmin, ymin, xmax, ymax)
        boxes = tf.concat([box_xmin, box_ymin, box_xmax, box_ymax], axis=-1)
        #confidence
        box_confidence = tf.sigmoid(x[..., 4:5])
        #classes
        box_class_prob = tf.sigmoid(x[..., 5:])

        output = tf.concat([boxes, box_confidence, box_class_prob], axis=-1)
        output = tf.reshape(output, [-1, self.grid_shape[0]*self.grid_shape[1]*self.num_anc, (5+self.num_cls)])
        return output
    
    def build(self, input_shape):
        #形成grid參數
        self.grid_shape = input_shape[1:3]
        self.grid = tf.meshgrid(tf.range(self.grid_shape[1]), tf.range(self.grid_shape[0]))
        self.grid = tf.expand_dims(tf.stack(self.grid, axis=-1), axis=2)
        self.grid = tf.cast(self.grid, tf.float32)


def creat_CSPYOLO(**kwargs):
    #input shape(h, w)
    input_shape = (kwargs['image_shape'][0], kwargs['image_shape'][1], 3)
    #類別總數
    num_classes = kwargs['num_classes']
    #將anchors 歸一化
    anchors = np.array(kwargs['anchors'])
    anchors_mask = kwargs['anchors_mask']
    #每一層輸出的anchors數量
    anc_pre_l = len(anchors) // 3

    input_layer = tf.keras.Input(shape=input_shape)
    ######Tiny-Backbone#####
    x = unit_conv(tensor=input_layer, num_filters=32, k_size=3, strides=2)
    block_1, _ = CSP_Block(x, num_filters=64, first=True)
    block_2, _ = CSP_Block(block_1, num_filters=128)
    block_3, feature = CSP_Block(block_2, num_filters=256)
    x = tf.keras.layers.MaxPooling2D(2, 2, 'same')(block_3)
    x = unit_conv(tensor=x, num_filters=512, k_size=3, strides=1)

    ######detector#####
    x = unit_conv(tensor=x, num_filters=256, k_size=1, strides=1)
    output_1 = unit_conv(tensor=x, num_filters=512, k_size=3, strides=1)
    output_1 = unit_conv(tensor=output_1, num_filters=255, k_size=1, strides=1, use_bn=False)
    output_1 = yolo_head(anchors=anchors[anchors_mask[1]], num_cls=num_classes, img_shape=input_shape)(output_1)

    #upsample
    output_2 = unit_conv(tensor=x, num_filters=128, k_size=1, strides=1)
    output_2 = tf.keras.layers.UpSampling2D()(output_2)
    output_2 = tf.keras.layers.Concatenate()([output_2, feature])
    output_2 = unit_conv(tensor=output_2, num_filters=256, k_size=3, strides=1)
    output_2 = unit_conv(tensor=output_2, num_filters=255, k_size=1, strides=1, use_bn=False)
    output_2 = yolo_head(anchors=anchors[anchors_mask[0]], num_cls=num_classes, img_shape=input_shape)(output_2)

    return tf.keras.Model(inputs=input_layer, outputs=[output_1, output_2])

def output_result(y_pred, **kwargs):
    '''
    Args:
        y_pred: model output (one image), list like: [[batch, neck_3_out], [batch, neck_2_out], [batch, neck_1_out]]
    '''
    num_box = []
    num_cls = []
    num_score = []
    for pred_out in y_pred:
        pred_out = tf.reshape(pred_out, [-1, 5 + kwargs['num_classes']])
        boxes = tf.concat([pred_out[...,1:2],
                           pred_out[...,0:1],
                           pred_out[...,3:4],
                           pred_out[...,2:3]], axis=-1)
        box_conf = pred_out[...,4:5]
        box_cls = pred_out[...,5:]
        mask = tf.reshape((box_conf > kwargs['score_threshold']), (-1,))
        box_conf = tf.boolean_mask(box_conf, mask)
        box_cls = tf.boolean_mask(box_cls, mask)
        boxes = tf.boolean_mask(boxes, mask)

        num_box.extend(boxes)
        num_cls.extend(box_cls)
        num_score.extend(box_conf)
    
    num_box = np.array(num_box)
    num_cls = np.array(num_cls)
    num_score = np.array(num_score)

    if len(num_box) == 0:
        return num_box, num_cls, num_score
    else:
        #暫時使用普通NMS
        nms_index = tf.image.non_max_suppression(num_box, tf.reshape(num_score, (-1,)), 40, iou_threshold=kwargs['iou_threshold'])
        select_boxes = tf.gather(num_box, nms_index)
        select_cls = tf.math.argmax(tf.gather(num_cls, nms_index), -1)
        select_score = tf.gather(num_score, nms_index)

        return select_boxes, select_cls, select_score

'''
load weights:
reference: https://github.com/hunglc007/tensorflow-yolov4-tflite
'''
def load_weights(model, weights_file, custom_cls=False):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    j = 0
    for i in range(21):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        #output layer
        if i not in [17, 20]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            filters = 255
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [17, 20]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            if not(custom_cls):
                conv_layer.set_weights([conv_weights, conv_bias])
    assert len(wf.read()) == 0, 'failed to read all data'
    print("load OK")
    wf.close()