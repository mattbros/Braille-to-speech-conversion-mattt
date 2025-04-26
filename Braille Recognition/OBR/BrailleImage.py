import cv2
import numpy as np

class BrailleImage(object):
    def __init__(self, image):
        # Read source image
        self.original = cv2.imread(image)
        if self.original is None:
            raise IOError('Cannot open given image')

        # First Layer, Convert BGR(Blue Green Red Scale) to Gray Scale
        gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        # Save the binary image of the edge detected
        self.edged_binary_image = self.__get_edged_binary_image(gray)

        # Now do the same to save a binary image to get the contents
        # inside the edges to see if the dot is really filled.
        self.binary_image = self.__get_binary_image(gray)
        self.final = self.original.copy()
        self.height, self.width, self.channels = self.original.shape
        return;

    def bound_box(self, left, right, top, bottom, color=(255, 0, 0), size=1):
        self.final = cv2.rectangle(self.final, (left, top), (right, bottom), color, size)
        return True

    def get_final_image(self):
        return self.final

    def get_original_image(self):
        return self.original

    def get_edged_binary_image(self):
        return self.edged_binary_image

    def get_binary_image(self):
        return self.binary_image

    def get_height(self):
        return self.height

    def get_width(self):
        return self.width

    def __get_edged_binary_image(self, gray):
        # First Lvl Blur to Reduce Noise - Even more aggressive and adaptive blurring
        blur = cv2.GaussianBlur(gray, (7, 7), 0)  # Further increased kernel size

        # Adaptive Thresholding to define the dots in Braille - More adaptive parameters
        thres = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            19,  # Increased block size even more
            -3  # Slightly more negative C
        )
        # Remove more Noise from the edges.
        blur2 = cv2.medianBlur(thres, 5) # Increased median blur
        # Sharpen again.
        ret2, th2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Remove more Noise.
        blur3 = cv2.GaussianBlur(th2, (5, 5), 0)
        # Final threshold
        ret3, th3 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(th3)

    def __get_binary_image(self, gray):
        blur = cv2.GaussianBlur(gray, (7, 7), 0)  # Increased kernel size
        ret2, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur2 = cv2.medianBlur(th2, 5) # Increased median blur
        ret3, th3 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(th3)
    
    def __get_edged_binary_image(self, gray):
        # First Lvl Blur to Reduce Noise - Even more aggressive and adaptive blurring
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        # Adaptive Thresholding to define the dots in Braille - More adaptive parameters
        thres = self.adaptive_threshold_with_local_stats(blur, 19, -0.5)
        #thres = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, -3)

        # Remove more Noise from the edges.
        blur2 = cv2.medianBlur(thres, 5)
        # Sharpen again.
        ret2, th2 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Remove more Noise.
        blur3 = cv2.GaussianBlur(th2, (5, 5), 0)
        # Final threshold
        ret3, th3 = cv2.threshold(blur3, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(th3)

    def __get_binary_image(self, gray):
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        ret2, th2 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        blur2 = cv2.medianBlur(th2, 5)
        ret3, th3 = cv2.threshold(blur2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.bitwise_not(th3)
    
    def adaptive_threshold_with_local_stats(self, image, block_size, k):
        mean, stddev = cv2.meanStdDev(image)
        C = mean - k * stddev
        return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, block_size, C)
    
    def __get_valid_dots(self, circles, binary_image, original_image):
        valid_dots = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                #print(f"Circle: x={x}, y={y}, r={r}")  # Debugging
                # Inside the circle, check if the majority of the pixels are white (dot)
                mask = np.zeros_like(binary_image)
                cv2.circle(mask, (x, y), r, 255, -1)  # Fill the circle
                white_pixels = np.sum(binary_image[y - r:y + r, x - r:x + r] & mask[y - r:y + r, x - r:x + r])
                total_pixels = np.sum(mask[y - r:y + r, x - r:x + r])
                #print(f"White pixels: {white_pixels}, Total pixels: {total_pixels}") # Debugging
                if total_pixels > 0 and white_pixels / total_pixels > 0.6:  # Adjust threshold as needed
                    valid_dots.append((x, y, r))
                    cv2.circle(original_image, (x, y), r, (0, 255, 0), 2)  # Mark valid
                else:
                    cv2.circle(original_image, (x, y), r, (255, 0, 0), 2)  # Mark invalid
        return valid_dots
