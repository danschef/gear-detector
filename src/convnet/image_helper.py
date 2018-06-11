from PIL import Image

def save_image(idx, tensor):
    npimg = tensor.numpy()
    image = Image.fromarray(npimg, 'RGBA')
    image.save(f"./imagery/prediction/aquafarm_{idx}.tif", 'TIFF')
