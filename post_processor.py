import shapely
from constants import HOT_ZONE_H, HOT_ZONE_W

class PostProcessor:
    def run(self, boxes):
        shapely_boxes = []

        for box in boxes:
            x, _y, w, _h = box

            shapely_boxes.append(shapely.box(x, 0, x + w, HOT_ZONE_H))

        combined_box = shapely.union_all(shapely_boxes)

        hot_zone = shapely.box(0, 0, HOT_ZONE_W, HOT_ZONE_H)
        hot_zone_intersection = shapely.intersection(combined_box, hot_zone)
        hot_zone_percentage = round(hot_zone_intersection.area / (HOT_ZONE_H * HOT_ZONE_W) * 100, 2)

        print(f"Hot zone {hot_zone_percentage}% full")

        return hot_zone_intersection
