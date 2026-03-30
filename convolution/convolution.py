import cv2
import numpy as np
from pathlib import Path



def find_image(candidates):
    for name in candidates:
        p = Path(name)
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"이미지를 찾지 못했습니다. 후보 파일명: {candidates}")



def save_image(path, image):
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path}")



def main():
    gray = cv2.imread("./input/grayscale.jpg", cv2.IMREAD_GRAYSCALE)
    binary = cv2.imread("./input/binary.png", cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)


    kernel = np.ones((3, 3), dtype=np.uint8)


    gray_dilation = cv2.dilate(gray, kernel, iterations=1)
    gray_erosion = cv2.erode(gray, kernel, iterations=1)


    binary_dilation = cv2.dilate(binary, kernel, iterations=1)
    binary_erosion = cv2.erode(binary, kernel, iterations=1)


    output_dir = Path('output_images')
    output_dir.mkdir(exist_ok=True)
    save_image(output_dir / 'gray_dilation.png', gray_dilation)
    save_image(output_dir / 'gray_erosion.png', gray_erosion)
    save_image(output_dir / 'binary_dilation.png', binary_dilation)
    save_image(output_dir / 'binary_erosion.png', binary_erosion)


    print('저장 완료:')
    for name in ['gray_dilation.png', 'gray_erosion.png', 'binary_dilation.png', 'binary_erosion.png']:
        print(output_dir / name)


if __name__ == '__main__':
    main()