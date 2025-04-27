import cv2
import numpy as np
from math import sqrt
from collections import Counter
from .BrailleCharacter import BrailleCharacter

class SegmentationEngine(object): 
    def __init__(self, image = None):
        self.image = image
        self.initialized = False
        self.dots = []
        self.diameter = 0.0
        self.radius = 0.0
        self.next_epoch = 0
        self.characters = []
        return

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if not self.initialized:
            self.initialized = True
            contours = self.__process_contours()
            if len(contours) == 0:
                # Since we have no dots.
                self.__clear()
                raise StopIteration()
            enclosingCircles = self.__get_min_enclosing_circles(contours)
            if len(enclosingCircles) == 0:
                self.__clear()
                raise StopIteration()

            diameter,dots,radius = self.__get_valid_dots(enclosingCircles)
            if len(dots) == 0:
                self.__clear()
                raise StopIteration()
            self.diameter = diameter
            self.dots = dots
            self.radius = radius
            self.next_epoch = 0
            self.characters = []

        if len(self.characters) > 0:
            r = self.characters[0]
            del self.characters[0]
            return r

        cor = self.__get_row_cor(self.dots, epoch=self.next_epoch) # do not respect breakpoints
        if cor is None:
            self.__clear()
            raise StopIteration()

        top = int(cor[1] - int(self.radius*1.5)) # y coordinate
        self.next_epoch = int(cor[1] + self.radius)

        cor = self.__get_row_cor(self.dots,self.next_epoch,self.diameter,True)
        if cor is None:
            # Assume next epoch
            self.next_epoch = int(self.next_epoch + (2*self.diameter))
        else:
            self.next_epoch = int(cor[1] + self.radius)

        cor = self.__get_row_cor(self.dots,self.next_epoch,self.diameter,True)
        if cor is None:
            self.next_epoch = int(self.next_epoch + (2*self.diameter))
        else:
            self.next_epoch = int(cor[1] + self.radius)
        
        bottom = self.next_epoch
        self.next_epoch += int(2*self.diameter)

        DOI = self.__get_dots_from_region(self.dots, top, bottom)
        xnextEpoch = 0
        while True:
            xcor = self.__get_col_cor(DOI, xnextEpoch)
            if xcor is None:
                break

            left = int(xcor[0] - self.radius) # x coordinate
            xnextEpoch = int(xcor[0] + self.radius)
            xcor = self.__get_col_cor(DOI,xnextEpoch,self.diameter,True)
            if xcor is None:
                # Assumed
                xnextEpoch += int(self.diameter*1.5)
            else:
                xnextEpoch = int(xcor[0]) + int(self.radius)
            right = xnextEpoch
            box = (left, right, top, bottom)
            dts = self.__get_dots_from_box(DOI, box)
            char = BrailleCharacter(dts, self.diameter, self.radius, self.image)
            char.left = left
            char.right = right
            char.top = top
            char.bottom = bottom
            self.characters.append(char)

        if len(self.characters) < 1:
            self.__clear()
            raise StopIteration()

        r = self.characters[0]
        del self.characters[0]
        return r

    def __clear(self):
        self.image = None
        self.initialized = False
        self.dots = []
        self.diameter = 0.0
        self.radius = 0.0
        self.next_epoch = 0
        self.characters = []

    def update(self, image):
        self.__clear()
        self.image = image
        return True

    def __get_row_cor(self, dots, epoch = 0, diameter = 0, respectBreakpoint = False):
        if len(dots) == 0:
            return None
        minDot = None
        for dot in dots:
            x,y = dot[0]
            if y < epoch:
                continue

            if minDot is None:
                minDot = dot
            else:
                v = int(y - epoch)
                minV = int(minDot[0][1] - epoch)
                if minV > v:
                    minDot = dot
                else:
                    continue
        if minDot is None:
            return None
        if respectBreakpoint:
            v = int(minDot[0][1] - epoch)
            if v > (2*diameter):
                return None # indicates that the entire row is not set
        return minDot[0] # (X,Y)

    def __get_col_cor(self, dots, epoch = 0, diameter = 0, respectBreakpoint = False):
        if len(dots) == 0:
            return None
        minDot = None
        for dot in dots:
            x,y = dot[0]
            if x < epoch:
                continue

            if minDot is None:
                minDot = dot
            else:
                v = int(x - epoch)
                minV = int(minDot[0][0] - epoch)
                if minV > v:
                    minDot = dot
                else:
                    continue
        if minDot is None:
            return None
        if respectBreakpoint:
            v = int(minDot[0][0] - epoch)
            if v > (2*diameter):
                return None # indicates that the entire row is not set
        return minDot[0] # (X,Y)

    def __get_dots_from_box(self, dots, box):
        left,right,top,bottom = box
        result = []
        for dot in dots:
            x,y = dot[0]
            if x >= left and x <= right and y >= top and y <= bottom:
                result.append(dot)
        return result

    def __get_dots_from_region(self, dots, y1, y2):
        D = []
        if y2 < y1:
            return D

        for dot in dots:
            x,y = dot[0]
            if y > y1 and y < y2:
                D.append(dot)
        return D

    def __get_valid_dots(self, circles):
        tolerance = 0.45
        dots = []
        local_radii = []  # Store radii for each region
        image_height = self.image.get_height()
        image_width = self.image.get_width()
        num_regions_y = 3  # Divide image vertically into 3 regions (adjust as needed)
        region_height = image_height // num_regions_y
        min_dist_between_dots = 0.7 #tune
        
        for i in range(num_regions_y):
            y_start = i * region_height
            y_end = (i + 1) * region_height
            region_circles = [
                c for c in circles if y_start <= c[0][1] < y_end
            ]  # Circles in this region
            
            region_radii = [c[1] for c in region_circles]
            if region_radii:
                local_baserad = Counter(region_radii).most_common(1)[0][0]
            else:
                local_baserad = 0
            
            for circle in region_circles:
                x, y = circle[0]
                rad = circle[1]
                if local_baserad > 0 and rad <= int(local_baserad * (1 + tolerance)) and rad >= int(local_baserad * (1 - tolerance)):
                    dots.append(circle)
                    local_radii.append(rad)  # Store the radius used for filtering
        
        if local_radii:
            baserad = Counter(local_radii).most_common(1)[0][0] # Overall radius
        else:
            baserad = 0
        
        valid_dots = []
        if baserad > 0:
            for dot in dots:
                is_duplicate = False
                for existing_dot in valid_dots:
                    dist = sqrt(((dot[0][0] - existing_dot[0][0]) ** 2) + ((dot[0][1] - existing_dot[0][1]) ** 2))
                    if dist < min_dist_between_dots * baserad:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    valid_dots.append(dot)
        
        return 2 * (baserad), valid_dots, baserad

    def __get_min_enclosing_circles(self, contours):
        circles = []
        for contour in contours:
            (x,y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            circles.append((center, radius))
        return circles

    def __process_contours(self):
        edg_bin_img = self.image.get_edged_binary_image()
        contours = cv2.findContours(edg_bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 2:
            contours = contours[0]
        else:
            contours = contours[1]
        return contours
