def natural_sort_key(file_name):
    import re
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', str(file_name))]

def load_images(img_dir):
    from pathlib import Path
    from PIL import Image
    paths = sorted(list(Path(img_dir).glob("*.png")), key=natural_sort_key)
    #print(paths)
    images = []
    for img_path in paths:
        images.append(Image.open(img_path))
    return images, paths