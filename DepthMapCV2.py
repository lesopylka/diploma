import OpenEXR
import Imath
import numpy as np
import cv2
import torch
from torchvision.transforms import Compose, ToTensor, Resize
from PIL import Image 

exr_path = r"C:/Users/missm/OneDrive/Документы/Unreal Projects/diploma/mvp/Saved/Screenshots/WindowsEditor/scene_render.exr"
png_path = "scene_render_half_float.png"
depth_map_path = "depth_map.png"

def read_exr_half_float(file_path):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    channels = ['R', 'G', 'B']  
    img_data = {}

    for channel in channels:
        raw_data = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.HALF))  
        img_data[channel] = np.frombuffer(raw_data, dtype=np.float16).reshape((height, width))

    return img_data

image_data = read_exr_half_float(exr_path)

if image_data:
    print("EXR-файл успешно прочитан")

    rgb_image = np.stack([image_data['R'], image_data['G'], image_data['B']], axis=-1)

    rgb_image = np.nan_to_num(rgb_image) 
    rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

    cv2.imwrite(png_path, rgb_image)
    print(f"EXR конвертирован в PNG: {png_path}")
else:
    print("Ошибка: Не удалось загрузить EXR")
    exit()

try:
    print("Загружаем MiDaS для обработки глубины...")
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas.eval()
    print("MiDaS загружена успешно.")
except Exception as e:
    print(f"Ошибка при загрузке модели MiDaS: {e}")
    exit()

def process_image(image_path):
    transform = Compose([Resize((384, 384)), ToTensor()])
    
    img = cv2.imread(image_path)
    if img is None:
        print("Ошибка: OpenCV не может открыть PNG!")
        return

    print("PNG загружен успешно:", img.shape)

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        depth = midas(img)

    depth = depth.squeeze().numpy()
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = np.uint8(depth)

    try:
        cv2.imwrite(depth_map_path, depth)
        print(f"Глубинная карта сохранена: {depth_map_path}")
    except Exception as e:
        print(f"Ошибка при сохранении глубинной карты: {e}")


process_image(png_path)
