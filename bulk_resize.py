import cv2
import os
import sys
from glob import glob

WIDTH = 96
HEIGHT = 96


def bulk_resize(src, dst):
    files = [y for x in os.walk(src) for y in glob(os.path.join(x[0], '*.*'))]
    for image_file in files:
        target_path = "/".join(image_file.strip("/").split('/')[1:-1])
        target_path = os.path.join(dst, target_path) + "/"
        if not os.path.exists(target_path):
            os.makedirs(target_path)
        image = cv2.imread(image_file)
        resized_image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        filename = os.path.basename(image_file).split('.')[0]
        cv2.imwrite(
            os.path.join(target_path, filename + ".jpg"),
            resized_image
        )

def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: bulk_resize.py <source-dir> <target-dir>\n")
        sys.exit(-1)

    bulk_resize(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
