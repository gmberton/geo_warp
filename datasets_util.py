
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


def open_image_and_apply_transform(image_path):
    """Given the path of an image, open the image, and return it as a normalized tensor.
    """
    
    pil_image = Image.open(image_path)
    tensor_image = transform(pil_image)
    return tensor_image
