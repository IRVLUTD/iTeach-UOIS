import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class BoundingBoxAnnotator:
    def __init__(self, image):
        self.image = image
        self.bboxes = []
        self.start_point = None
        self.current_rect = None

    def on_press(self, event):
        if event.inaxes is not None:
            self.start_point = (event.xdata, event.ydata)

    def on_release(self, event):
        if event.inaxes is not None and self.start_point is not None:
            x1, y1 = self.start_point
            x2, y2 = event.xdata, event.ydata

            # Compute bounding box (cx, cy, w, h)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = abs(x2 - x1), abs(y2 - y1)

            self.bboxes.append((cx, cy, w, h))
            self.start_point = None

            # Draw the rectangle
            self.redraw()

    def redraw(self):
        plt.clf()
        plt.imshow(self.image)
        ax = plt.gca()
        for (cx, cy, w, h) in self.bboxes:
            rect = plt.Rectangle((cx - w/2, cy - h/2), w, h, edgecolor='r', facecolor='none', lw=2)
            ax.add_patch(rect)

        plt.draw()

    def on_key(self, event):
        if event.key == 'q':  # Quit and process the bounding boxes
            plt.close()
            print("Bounding boxes collected:", self.bboxes)
            # Continue with the rest of the code here

    def start(self):
        fig, ax = plt.subplots()
        ax.imshow(self.image)
        fig.canvas.mpl_connect('button_press_event', self.on_press)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        fig.canvas.mpl_connect('key_press_event', self.on_key)
        plt.show()
        return self.bboxes  # Return collected bounding boxes

# Usage
def main():
    image_path = "/home/jishnu/iTeach-UOIS-Data-Collection/data/scene6/jpg/000000.jpg"  # Change this to your image path
    image = Image.open(image_path)
    
    annotator = BoundingBoxAnnotator(image)
    bounding_boxes = annotator.start()

    # Continue processing with bounding_boxes
    print("Final bounding boxes:", bounding_boxes)

if __name__ == "__main__":
    main()
