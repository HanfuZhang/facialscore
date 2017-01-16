import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2, dlib
import math
import argparse
import pdb

CLASSES = ('__background__', '1')

NETS = {'vgg16': ('VGG16', 'facedetection.caffemodel')}

PATH = '/home/deepir/facialscore/'

image_list = PATH + 'data/face_score.txt'

LANDMARKS = dlib.shape_predictor(PATH + 'data/shape_predictor_68_face_landmarks.dat')

SHOW = 0 

def detectFaces(net,img):
# detect the faces
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net,img)
    timer.toc()
    print ('Detection took {:.3f}s').format(timer.total_time)
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    result = []
    mark = 0
    new_mark = 0
    (h,w) = img.shape[:2]
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
                print bbox
                if int(bbox[0]) > 0 and int(bbox[1]) > 0 and int(bbox[2]) < w and int(bbox[3]) < h:
                    bbox[1] = 0.8*bbox[1] + 0.2*bbox[3] 
                    if SHOW == 1:
                        print SHOW
                        cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,255,0),1)
                        cv2.imshow('image',img)
                        cv2.waitKey(0)
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
                    new_mark = detectNewImg(img,eyes)
    return result, new_mark 

def detectNewImg(img,eyes):
    if SHOW == 1:
        plt.figure(1)
        for i in range(68):
            plt.plot(eyes.part(i).x,eyes.part(i).y,'ro')
        plt.imshow(img)
    new_mark = 0
    (h,w) = img.shape[:2]
    minx = min(eyes.part(i).x for i in range(68))
    maxx = max(eyes.part(i).x for i in range(68))
    miny = min(eyes.part(i).y for i in range(68))
    maxy = max(eyes.part(i).y for i in range(68))
    if minx > 0 and miny > 0 and maxx < w and maxy < h:
        left = eyes.part(30).x - minx
        right = maxx - eyes.part(30).x
        if abs(left - right) < 40:
            new_mark = 1
    return new_mark

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

    data = open('/home/deepir/hanfu.json','r').readlines():
        img = data[i].split('"')[25] + '/1.jpg'
        faces, mark = detectFaces(net,img)
        if mark == 1 and len(faces) == 1:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurryscore = cv2.Laplacian(gray, cv2.CV_64F).var()
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            mean_y = np.mean(ycrcb[:,:,0])

            if blurryscore > 200 and mean_y > 100:
                for (x1,y1,x2,y2) in faces:
                    if float((x2 - x1) * (y2 - y1))/float(448 * 256) > 0.2:
                        face = img[y1:y2,x1:x2]
                        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        blurryscore_face = cv2.Laplacian(gray_face, cv2.CV_64F).var()
                        if blurryscore_face > 150:
                            os.system('cp ' + PATH + 'data/yanzhi/images/' + path.split()[0] + ' ' + PATH + 'data/yanzhi/newimages/' + path.split()[0])
