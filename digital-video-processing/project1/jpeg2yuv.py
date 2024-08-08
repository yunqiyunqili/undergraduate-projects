from PIL import Image
import numpy as np

def rgb_to_yuv(rgb_image):
    # Convert RGB image to YUV format
    rgb_image = np.array(rgb_image, dtype=np.float32) / 255.0
    R = rgb_image[:, :, 0]
    G = rgb_image[:, :, 1]
    B = rgb_image[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.614 * R - 0.51498 * G - 0.10001 * B

    YUV = np.stack([Y, U, V], axis=-1)
    return YUV

def save_yuv_image(yuv_image, filename):
    # Save YUV image to a file
    height, width, _ = yuv_image.shape
    yuv_image = (yuv_image * 255).astype(np.uint8)
    
    y = yuv_image[:, :, 0]
    u = yuv_image[:, :, 1] + 128  # Shifting U to the range [0, 255]
    v = yuv_image[:, :, 2] + 128  # Shifting V to the range [0, 255]

    with open(filename, 'wb') as f:
        f.write(y.tobytes())
        f.write(u.tobytes())
        f.write(v.tobytes())

def jpeg_to_yuv(jpeg_filename, yuv_filename):
    # Load JPEG image
    rgb_image = Image.open(jpeg_filename).convert('RGB')
    
    # Convert to YUV
    yuv_image = rgb_to_yuv(np.array(rgb_image))
    
    # Save YUV image
    save_yuv_image(yuv_image, yuv_filename)

if __name__ == "__main__":
    jpeg_filename = 'lena_noise.jpeg'
    yuv_filename = 'lena_noise.yuv'
    jpeg_to_yuv(jpeg_filename, yuv_filename)
    print(f'Converted {jpeg_filename} to {yuv_filename}')
