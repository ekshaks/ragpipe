from pathlib import Path

def natural_sort_key(file_name):
    import re
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', str(file_name))]

def get_sorted_paths(img_dir, ext="png"):
    paths = sorted(list(Path(img_dir).glob(f"*.{ext}")), key=natural_sort_key)
    return paths

def load_images(img_dir, format='png'):
    from PIL import Image
    paths = sorted(list(Path(img_dir).glob(f"*.{format}")), key=natural_sort_key)
    #print(paths)
    images = []
    for img_path in paths:
        images.append(Image.open(img_path))
    return images, paths