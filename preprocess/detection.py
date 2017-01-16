import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2, dlib
import json
import math
import argparse
import pdb

CLASSES = ('__background__', '1')

NETS = {'vgg16': ('VGG16', 'facedetection.caffemodel')}

PATH = '/home/deepir/facialscore/'

LANDMARKS = dlib.shape_predictor(PATH + 'data/shape_predictor_68_face_landmarks.dat')

SHOW = 0

def detectFaces(net,img):
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net,img)
    timer.toc()
    print ('Detection took {:.3f}s').format(timer.total_time)
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    mark = 0
    result = []
    new_img = img
    new_result = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1
        cls_boxes = boxes[:,4*cls_ind:4*(cls_ind+1)]
        cls_scores = scores[:,cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:,np.newaxis])).astype(np.float32)
        keep = nms(dets,NMS_THRESH)
        dets = dets[keep,:]
        inds = np.where(dets[:,-1]>=CONF_THRESH)[0]
        print 'inds.size',inds.size
        if len(inds) != 0:
            mark = 1
            for j in inds:
                bbox = dets[j,:4]
                bbox[1] = 0.8*bbox[1] + 0.2*bbox[3] 
                if SHOW == 1:
                    cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),1)
                    cv2.imshow('image',img)
                result.append((int(bbox[0]),int(bbox[1]),int(bbox[2]),int(bbox[3])))
        else:
            temp = dlib.get_frontal_face_detector()(img,1)
            if len(temp) != 0:
                mark = 1
                temp_face = max(temp, key = lambda rect: rect.width() * rect.height())
                result.append((temp_face.left(),temp_face.top(),temp_face.right(),temp_face.bottom()))
        if mark == 1:
            if len(result) == 1:
                for (x1,y1,x2,y2) in result:
                    sub_faces = dlib.rectangle(left = x1, top = y1, right = x2, bottom = y2)
                    eyes = LANDMARKS(img,sub_faces)
                    new_img, result = detectNewImg(img,eyes,new_result)
    return new_img, new_result

def detectNewImg(img,eyes,result):
    tempx = 0
    tempy = 0
    if SHOW == 1:
        plt.figure(1)
        plt.subplot(121)
        for i in range(68):
            plt.plot(eyes.part(i).x,eyes.part(i).y,'ro')
        plt.imshow(img)
    for i in range(36,42):
        tempx += eyes.part(i).x - eyes.part(i + 6).x
        tempy += eyes.part(i).y - eyes.part(i + 6).y
    if tempx == 0:
        new_img = img
    else:
        theta = math.atan((float)(tempy) / (float)(tempx))
        center = (eyes.part(36).x, eyes.part(36).y)
        (h,w) = img.shape[:2]
        matrix = cv2.getRotationMatrix2D(center, theta*90, 1)
        new_img = cv2.warpAffine(img, matrix, (w,h))
        temp_x = [0 for i in range(68)]
        temp_y = [0 for i in range(68)]
        for i in range(68):
            temp_x[i] = (eyes.part(i).x - eyes.part(36).x)*math.cos(-theta) - (eyes.part(i).y - eyes.part(36).y)*math.sin(-theta) + eyes.part(36).x
            temp_y[i] = (eyes.part(i).x - eyes.part(36).x)*math.sin(-theta) + (eyes.part(i).y - eyes.part(36).y)*math.cos(-theta) + eyes.part(36).y
        if SHOW == 1:
            plt.subplot(122)
            for i in range(68):
                plt.plot(temp_x[i],temp_y[i],'ro')
            plt.imshow(new_img)
            plt.show()
    eye_y = sum(temp_y[36:48])/12
    mouth_y = sum(temp_y[48:68])/20
    ymin = eye_y - (mouth_y - eye_y)/30*45
    ymax = mouth_y + (mouth_y - eye_y)/30*25
    all_x = sum(temp_x)/68
    xmin = all_x - (mouth_y - eye_y)/30*50
    xmax = all_x + (mouth_y - eye_y)/30*50
    xmin = min(max(xmin, 0), w)
    xmax = min(max(xmax, 0), w)
    ymin = min(max(ymin, 0), h)
    ymax = min(max(ymax, 0), h)
    result.append(((int)(xmin),(int)(ymin),(int)(xmax),(int)(ymax)))
    return new_img, result

def parse_args():
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]', default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode', help='Use CPU mode (overrides --gpu)', action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]', choices=NETS.keys(), default='vgg16')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    prototxt = os.path.join(PATH, 'models/pascal_voc/VGG16', 'faster_rcnn_end2end', 'test.prototxt')
    caffemodel = os.path.join(PATH, 'data/faster_rcnn_models', NETS[args.demo_net][1])
    print prototxt
    print caffemodel
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\n').format(caffemodel))
    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\n\nLoaded network {:s}'.format(caffemodel)

    #annotation = open(PATH + 'data/image_list.json', 'r')
    #image_list = open(PATH + 'data/image_list.txt', 'w')
    #data = json.load(annotation)
    #pdb.set_trace()
    #for i in range(len(data)):
        #path = data[i]['path'].encode("utf-8")
        #label = data[i]['label'].encode("utf-8")
    
    #annotation = open('/home/deepir/facialscore/data/face_score.txt','r')
    #for path in annotation.readlines():
    path = '/home/public/inke' 
    for root, dirs, files in os.walk(path):
        for num, file in enumerate(files):
            print root + '/' + file, num
            img = cv2.imread(root + '/' + file)
            if img is not None:
                (H, W) = img.shape[:2]
                if H != W:
                    new_img, new_faces = detectFaces(net, img)
                    if len(new_faces) == 1:
                        for (x1,y1,x2,y2) in new_faces:
                            if SHOW == 1:
                                cv2.rectangle(new_img,(x1,y1),(x2,y2),(0,255,0),1)
                                cv2.imshow('image',new_img)
                                cv2.waitKey(0)
                            h = y2 - y1
                            w = x2 - x1
                            print h, w
                            if h != w:
                                maxx = max(h, w)
                                new_ = np.zeros((maxx, maxx, 3), np.uint8)
                                new_[(int)((maxx - h)/2):(int)((maxx + h)/2),(int)((maxx - w)/2):(int)((maxx + w)/2)] = new_img[y1:y2,x1:x2]
                            else:
                                new_ = new_img[y1:y2,x1:x2]
                            cv2.imwrite((root + '/' + file), new_)
                    else:
                        os.system('rm ' + root + '/' + file)
