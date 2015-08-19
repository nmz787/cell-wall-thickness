import cv2
import os
import sys
import numpy as np

class getThickness(object):
    WALL_EDGE_COLOR = (35,35,35)
    
    def __init__(self, img_path):
        self.img_path = os.path.abspath(img_path)
        self.img = cv2.imread(self.img_path)
        self.img_blank = cv2.im

        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        self.ret,self.thresh = cv2.threshold(imgray,127,255,0)
        self.im2, self.contours, self.hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    def adaptive_threshold(self):
        #img = cv2.imread('dave.jpg',0)
        img = cv2.medianBlur(self.img,5)
        
        ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
        
        titles = ['Original Image', 'Global Thresholding (v = 127)',
                    'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
        images = [img, th1, th2, th3]
        for img in images:
            cv2.imshow('image {}'.format(str(img)), img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, new_img_name='modified'):
        cv2.imwrite('{}_{}.png'.format(os.path.split(self.img_path)[0],
                                       new_img_name
                                       ),
                    self.img)

    def get_contours(self):
        # findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
        self.contours, self.hierarchy = cv2.findContours(self.img)

    def process_contours(self):
        for contour in self.contours:
            for point in contour:
                self.img_blank[point] = self.WALL_EDGE_COLOR




if __name__ == '__main__':
    if not len(sys.argv)>1:
        print 'usage: python cell_thickness.py <image path>'
    else:    
        img_path = sys.argv[1]
        gt = getThickness(img_path)
        gt.adaptive_threshold()
