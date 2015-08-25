import cv2
import os
import sys
import numpy as np

class getThickness(object):
    WALL_EDGE_COLOR = (35,35,35)
    
    def __init__(self, img_path):
        self.img_path = os.path.abspath(img_path)
        self.img = cv2.imread(self.img_path)
        #self.img = self.get_simple_concentric_circles()
        self.img = self.get_blue(self.img)

        cv2.imshow('', self.img)
        cv2.waitKey(0)
        self.get_contours(self.img)
        #return

        otsus = self.otsus_threshold(self.img)
        cv2.imshow("otsus", otsus)
        cv2.waitKey(0)

        #otsus_flooded = self.floodfill(otsus)

        cv2.destroyAllWindows()
        #self.get_contours(otsus_flooded)

        kernel = np.ones((5,5),np.uint8)
        # Opening is just another name of erosion followed by dilation. 
        # It is useful in removing noise, as we explained above. 
        # Here we use the function, cv2.morphologyEx()
        opening = cv2.morphologyEx(otsus, cv2.MORPH_OPEN, kernel)
        # Closing is reverse of Opening, Dilation followed by Erosion. 
        # It is useful in closing small holes inside the foreground objects,
        # or small black points on the object.
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        self.get_contours(closing)
        self.process_contours(closing)
        # print self.img.shape, type(self.img)
        # get the image properties, using the default of 1 channel for 2-axis images
        height, width, channels = self.get_image_dimensions(self.img)
        self.img_blank = np.zeros((height, width, 1), np.float64) # np.uint8

        imgray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY, 1)
        self.ret, self.thresh = cv2.threshold(imgray,127,255,0)
        #imgray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY, 1)
        # print type(self.ret), type(self.thresh)#, type(imgray)
        # print len(self.thresh)
        # print self.thresh.shape
        # print self.thresh.dtype

    def get_simple_concentric_circles(self):
        # Create a black image
        img = np.zeros((512,512,1), np.uint8)
        diameter = 63
        # origin is at upper left
        y=256
        x=256
        
        inner_diameter = 50
        outer_diameter = 150
        
        known_wall_thickness = (outer_diameter - inner_diameter)/2.0
        
        print 'known wall thickness: {}'.format(known_wall_thickness)

        # draw a big white circle
        cv2.circle(img,(x,y), outer_diameter, (255), -1)
        # draw a smaller black circle
        cv2.circle(img,(x,y), inner_diameter, (0), -1)

        return img

    def get_image_dimensions(self, img):
        return img.shape if len(img.shape)>2 else img.shape + (1,)

    def adaptive_threshold(self):
        #img = cv2.imread('dave.jpg',0)
        img = cv2.medianBlur(self.img,5)
        
        ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 1),255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                    cv2.THRESH_BINARY,11,2)
        th3 = cv2.adaptiveThreshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 1),255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
        
        

        images = [('Original Image'                , img),
                  ('Global Thresholding (v = 127)' , th1),
                  ('Adaptive Mean Thresholding'    , th2),
                  ('Adaptive Gaussian Thresholding', th3),
                  ('thresh',self.thresh),
                  ('imgret',self.imgret)]
        #images = [img, th1, th2, th3]
        for image_tuple in images:
            title, img = image_tuple
            cv2.imshow(title, img)
            cv2.waitKey(0)
        cv2.destroyAllWindows()

    def otsus_threshold(self, img):
        #from matplotlib import pyplot as plt
        
        #img = cv2.imread('noisy2.png',0)
        if self.get_image_dimensions(img)[2]>1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 1)

        # global thresholding
        ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
        
        # Otsu's thresholding
        ret2,th2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img,(5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print 'otsus dimensions: {}, dtype: {}'.format(th3.shape, th3.dtype)
        
        # plot all the images and their histograms
        images = [img, 0, th1,
                  img, 0, th2,
                  blur, 0, th3]
        titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
                  'Original Noisy Image','Histogram',"Otsu's Thresholding",
                  'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
        
        #for i in xrange(3):
        #    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        #    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        #    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        #    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        #    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        #    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
        #plt.show()
        return th3

    def save_image(self, new_img_name='modified'):
        cv2.imwrite('{}_{}.png'.format(os.path.split(self.img_path)[0],
                                       new_img_name
                                       ),
                    self.img)

    def get_blue(self, img):
        # Convert BGR to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
    
        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(img,img, mask= mask)
        return res


    def get_contours(self, img):
        print 'image to getContours is of type {} and shape {}'.format(img.dtype, self.get_image_dimensions(img))
        # convert a 3-channel to a single-channel if a color image was passed
        if self.get_image_dimensions(img)[2]>1:
            print 'converting pre-contour image to single-channel'
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY, 1)

        cv2.imshow("before contour_img", img)
        cv2.waitKey(0)
        # findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
        #self.contours, self.hierarchy = cv2.findContours(self.img)
        #self.contours, self.hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#cv2.CHAIN_APPROX_SIMPLE)
        imgret, contours0, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        self.contours = contours0

        c = max(contours0, key=cv2.contourArea)
 
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
         
        # draw a green bounding box surrounding the red game
        contour_img = img.copy()
        #cv2.drawContours(contour_img, [approx], -1, ( 255), 4)
        cv2.drawContours(contour_img, contours0, -1, ( 255,)*3, 4)
        cv2.imshow("contour_img", contour_img)
        cv2.waitKey(0)
        return contour_img

    def floodfill(self, img):
        print 'floodfill img shape: {}'.format(img.shape)
        h, w = img.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)
        self.seed_pt = None
        fixed_range = True
        connectivity = 4
        self.flooded = img.copy()
        
        def update(dummy=None):
            if self.seed_pt is None:
                #print 'seed_pt is None!!'
                cv2.imshow('floodfill', img)
                self.seed_pt = False
                return
            elif self.seed_pt is False:
                return
            #print 'seed_pt is: {}'.format(self.seed_pt)
            
            mask[:] = 0
            lo = cv2.getTrackbarPos('lo', 'floodfill')
            hi = cv2.getTrackbarPos('hi', 'floodfill')
            flags = connectivity
            if fixed_range:
                flags |= cv2.FLOODFILL_FIXED_RANGE
            cv2.floodFill(self.flooded, mask, self.seed_pt, (255, 255, 255), (lo,)*3, (hi,)*3, flags)
            #cv2.circle(self.flooded, self.seed_pt, 2, (0, 0, 255), -1)
            #cv2.destroyAllWindows()
            #cv2.waitKey(-1)
            #cv2.setMouseCallback('floodfill', onmouse)
            #cv2.createTrackbar('lo', 'floodfill', 20, 255, update)
            #cv2.createTrackbar('hi', 'floodfill', 20, 255, update)
            
            cv2.imshow('floodfill', self.flooded)
    
        def onmouse(event, x, y, flags, param):
            #print flags 
            #print '({}, {})'.format(x, y)
            #print [attr for attr in dir(cv2) if attr.startswith('EVENT_') and getattr(cv2, attr) & flags]
            if flags & cv2.EVENT_FLAG_LBUTTON:
                self.seed_pt = x, y
                update()
    
        update()
        cv2.setMouseCallback('floodfill', onmouse)
        cv2.createTrackbar('lo', 'floodfill', 20, 255, update)
        cv2.createTrackbar('hi', 'floodfill', 20, 255, update)
    
        while True:
            ch = 0xFF & cv2.waitKey()
            if ch == 27:
                break
            if ch == ord('f'):
                fixed_range = not fixed_range
                print 'using %s range' % ('floating', 'fixed')[fixed_range]
                update()
            if ch == ord('c'):
                connectivity = 12-connectivity
                print 'connectivity =', connectivity
                update()
        cv2.destroyAllWindows()
        return self.flooded


    def process_contours(self, img):
        contour_distance_maps = []
        for contour in self.contours:
            height, width, channels = self.get_image_dimensions(img)
            dist_map = np.zeros((height, width, 3), np.uint8) #np.float64) #
            for point in contour:
                # print [w for p in point for w in p]
                # print point
                # print point-2
                # print type(point)
                # print dir(point)
                # print ''
                # sys.exit()
                dist_map[point[0][1]][point[0][0]] = (255, 255, 255)#self.WALL_EDGE_COLOR
            contour_distance_maps.append(dist_map)
            cv2.imshow('contour processed', dist_map)
            cv2.waitKey(0)



if __name__ == '__main__':
    if not len(sys.argv)>1:
        print 'usage: python cell_thickness.py <image path>'
    else:    
        img_path = sys.argv[1]
        gt = getThickness(img_path)
        #gt.adaptive_threshold()
