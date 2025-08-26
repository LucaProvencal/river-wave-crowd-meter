import cv2

from detector import Detector
from renderer import Renderer
from stream import Stream
from post_processor import PostProcessor

def main():
    detector = Detector()
    post_processor = PostProcessor()
    renderer = Renderer()
    stream = Stream()

    image = stream.grab_frame_from_stream("https://www.youtube.com/watch?v=co-IVDGWtD8")

    boxes = detector.detect(image)
    hot_zone_intersection = post_processor.run(boxes)

    renderer.plot(boxes, hot_zone_intersection, image)

if __name__ == "__main__":
    main()
