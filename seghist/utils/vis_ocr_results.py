import json
from tqdm import tqdm
import os
from PIL import Image, ImageDraw,ImageFont

import urllib
import shutil

import mmcv
import cv2
import numpy as np

from seghist.utils.visualize import draw_polygons, is_contain_chinese, draw_texts, gen_color

def draw_texts_by_pil(img,
                      texts,
                      boxes=None,
                      draw_box=True,
                      on_ori_img=False,
                      font_size=None,
                      fill_color=None,
                      draw_pos=None,
                      return_text_size=False):
    """Draw boxes and texts on empty image, especially for Chinese.

    Args:
        img (np.ndarray): The original image.
        texts (list[str]): Recognized texts.
        boxes (list[list[float]]): Detected bounding boxes.
        draw_box (bool): Whether draw box or not. If False, draw text only.
        on_ori_img (bool): If True, draw box and text on input image,
            else on a new empty image.
        font_size (int, optional): Size to create a font object for a font.
        fill_color (tuple(int), optional): Fill color for text.
        draw_pos (list[tuple(int)], optional): Start point to draw each text.
        return_text_size (bool): If True, return the list of text size.

    Returns:
        (np.ndarray, list[tuple]) or np.ndarray: Return a tuple
        ``(out_img, text_sizes)``, where ``out_img`` is the output image
        with texts drawn on it and ``text_sizes`` are the size of drawing
        texts. If ``return_text_size`` is False, only the output image will be
        returned.
    """

    color_list = gen_color()
    h, w = img.shape[:2]
    if boxes is None:
        boxes = [[0, 0, w, 0, w, h, 0, h]]
    if draw_pos is None:
        draw_pos = [None for _ in texts]
    assert len(boxes) == len(texts) == len(draw_pos)

    if fill_color is None:
        fill_color = (0, 0, 0)

    if on_ori_img:
        out_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        out_img = Image.new('RGB', (w, h), color=(255, 255, 255))
    out_draw = ImageDraw.Draw(out_img)

    text_sizes = []
    for idx, (box, text, ori_point) in enumerate(zip(boxes, texts, draw_pos)):
        if len(text) == 0:
            continue
        min_x, max_x = min(box[0::2]), max(box[0::2])
        min_y, max_y = min(box[1::2]), max(box[1::2])
        color = tuple(list(color_list[idx % len(color_list)])[::-1])
        if draw_box:
            out_draw.line(box, fill=color, width=1)
        dirname = './utils/fonts'
        font_path = os.path.join(dirname, 'font.TTF')
        if not os.path.exists(font_path):
            url = ('https://download.openmmlab.com/mmocr/data/font.TTF')
            print(f'Downloading {url} ...')
            local_filename, _ = urllib.request.urlretrieve(url)
            shutil.move(local_filename, font_path)
        tmp_font_size = font_size
        if tmp_font_size is None:
            tmp_font_size = min(int(0.9 * (max_y - min_y) / len(text)), int((max_x - min_x) * 0.6))
        fnt = ImageFont.truetype(font_path, tmp_font_size)
        if ori_point is None:
            ori_point = (min_x + 1, min_y + 1)
        out_draw.text(ori_point, text, font=fnt, fill=fill_color, direction='ttb')

        text_sizes.append(fnt.getsize(text))

    del out_draw

    out_img = cv2.cvtColor(np.asarray(out_img), cv2.COLOR_RGB2BGR)

    if return_text_size:
        return out_img, text_sizes

    return out_img


def visualize_one_sample(img, boxes, texts):
    img = mmcv.imread(img)
    box_vis_img = draw_polygons(img, boxes)
    if is_contain_chinese(''.join(texts)):
        text_vis_img = draw_texts_by_pil(img, texts, boxes, draw_box=False, font_size=None)
    else:
        text_vis_img = draw_texts(img, texts, boxes, draw_box=False)

    h, w = img.shape[:2]
    out_img = np.ones((h * 2, w, 3), dtype=np.uint8)
    out_img[:h, :, :] = box_vis_img
    out_img[h:, :, :] = text_vis_img
    return out_img

def visualize(result_dir, image_dir, save_dir):
    image_list =  [f for f in os.listdir(image_dir) if not os.path.isdir(os.path.join(image_dir, f))]
    os.makedirs(save_dir, exist_ok=True)
    for res_fn, img_fn in tqdm(zip(sorted(os.listdir(result_dir)), 
                              sorted(image_list))):
        img_fp = os.path.join(image_dir, img_fn)
        with open(os.path.join(result_dir, res_fn)) as f:
            data_list = json.load(f)
        boxes, texts = [], []
        for data in data_list:
            boxes.append(data['polygon'])
            texts.append(data['text'])
        out_img = visualize_one_sample(img_fp, boxes, texts)
        save_fp = os.path.join(save_dir, img_fn)
        mmcv.imwrite(out_img, save_fp)


def main():
    result = '/home/huxingjian/model/mmocr/projects/PFRNet/project_samples/recog_results'
    image = '/home/huxingjian/model/mmocr/projects/PFRNet/project_samples'
    vis = '/home/huxingjian/model/mmocr/projects/PFRNet/project_samples/vis_results'
    visualize(result, image, vis)