import cv2

from detector import Detector
from renderer import Renderer
from post_processor import PostProcessor

def main():
    image = cv2.imread('assets/4.PNG')

    detector = Detector()
    renderer = Renderer()
    post_processor = PostProcessor()

    boxes = detector.detect(image)
    hot_zone_intersection = post_processor.run(boxes)

    renderer.plot(boxes, hot_zone_intersection, image)

if __name__ == "__main__":
    main()
