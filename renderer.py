import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from constants import HOT_ZONE_H, HOT_ZONE_W

class Renderer:
    def plot(self, boxes, shapely_geom, image):
        self._render_hot_zone(shapely_geom)
        self._render_boxes(boxes, image)

    def _render_hot_zone(self, shapely_geom):
        _, axs = plt.subplots()
        axs.set_aspect('equal', 'datalim')

        for geom in shapely_geom.geoms:    
            xs, ys = geom.exterior.xy    
            axs.fill(xs, ys, alpha=0.5, fc='r', ec='none')

        hot_zone_rect = patches.Rectangle((0, 0), HOT_ZONE_W, HOT_ZONE_H, alpha=0.2, fc='y', ec='none')
        axs.add_patch(hot_zone_rect)
        axs.invert_yaxis()

        plt.show()
    
    def _render_boxes(self, boxes, image):
        for box in boxes:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (math.floor(x + w), math.floor(y + h)), (0, 255, 0), 2)

        cv2.rectangle(
            image,
            (0, 0),
            (math.floor(0 + HOT_ZONE_W), math.floor(0 + HOT_ZONE_H)),
            (0, 255, 255),
            2
        )

        cv2.imshow('Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
