from PIL import Image
import os


def convert_jpg_to_png(source_directory, target_directory):

    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    for filename in os.listdir(source_directory):
        if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
            source_path = os.path.join(source_directory, filename)
            target_path = os.path.join(target_directory, f"{os.path.splitext(filename)[0]}.png")

            image = Image.open(source_path)
            image.save(target_path, "PNG")
            print(f"Converted and saved: {target_path}")

if __name__ == '__main__':
    source_dir = "./coco_seg/coco_images/"
    target_dir = "./coco_seg/coco_images_png/"
    convert_jpg_to_png(source_dir, target_dir)