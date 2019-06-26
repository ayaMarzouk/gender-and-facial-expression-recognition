import argparse
import cv2
import numpy as np
def yolo(Image_path):
    ap = argparse.ArgumentParser()
    
    ap.add_argument('--image',default=Image_path,
                    help = 'path to input image')
    ap.add_argument('--config',default='yolo/yolo.cfg',
                    help = 'path to yolo config file')
    ap.add_argument('--weights',default='yolo/yolo.weights',
                    help = 'path to yolo pre-trained weights')
    ap.add_argument('--classes', default='yolo/yolo.txt',
                    help = 'path to text file containing class names')
    args = ap.parse_args()


    def get_output_layers(net):
        layer_names = net.getLayerNames()

        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        return output_layers


    def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(classes[class_id])

        color = COLORS[class_id]

        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(img, str('{:.4f}'.format(confidence)), (x - 25, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    image = cv2.imread(args.image)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    #COLORS = np.random.uniform(0, 255, size=(len(classes), 1))
    COLORS=[[0,255,0]]
    #print(COLORS)
    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
    #image : This is the input image we want to preprocess before passing it through our deep neural network for classification.
    #scalefactor : After we perform mean subtraction we can optionally scale our images by some factor. This value defaults to 1.0 (i.e., no scaling) but we can supply another value as well. It’s also important to note that scalefactor  should be 1 / \sigma as we’re actually multiplying the input channels (after mean subtraction) by scalefactor .
    #size : Here we supply the spatial size that the Convolutional Neural Network expects. For most current state-of-the-art neural networks this is either 224×224, 227×227, or 299×299.
    #mean : These are our mean subtraction values. They can be a 3-tuple of the RGB means or they can be a single value in which case the supplied value is subtracted from every channel of the image. If you’re performing mean subtraction, ensure you supply the 3-tuple in (R, G, B) order, especially when utilizing the default behavior of swapRB=True .
    #swapRB : OpenCV assumes images are in BGR channel order; however, the mean value assumes we are using RGB order. To resolve this discrepancy we can swap the R and B channels in image  by setting this value to True. By default OpenCV performs this channel swapping for us.

    #returns a  blob  which is our input image after mean subtraction, normalizing, and channel swapping
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4 #allowed value of overlapping boxes ,kol ma ktarto hyzwod el overlap m3a t7yat omar medhat

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence >= 0.1:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    #print(len(boxes))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    Boundry_Boxes=[]
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = int(box[0]) - 45
        y = int(box[1]) - 45
        w = box[2] + 65
        h = box[3] + 65
        Boundry_Boxes.append([x,y,w,h])
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

    return Boundry_Boxes
