import OpenEXR
import Imath
import numpy as np
import cv2
import OpenImageIO as oiio

exr_path = r"C:/Users/missm/OneDrive/Документы/Unreal Projects/diploma/mvp/Saved/Screenshots/WindowsEditor/scene_render.exr"

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
    rgb_image = np.clip(rgb_image * 255, 0, 255).astype(np.uint8)

    cv2.imwrite("scene_render_half_float.png", rgb_image)
    print("Изображение сохранено как scene_render_half_float.png")
else:
    print("Ошибка: Не удалось загрузить EXR")
