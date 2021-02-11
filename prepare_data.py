"""This module creates crops of tables from the document images,
essentially used for data preparation"""

import os
import glob
import string
import pickle
import argparse
from xml.etree import ElementTree

import cv2
import numpy as np
import pytesseract
from PIL import Image

from truthpy.Document import Document
from augmentation.augmentor import translate_ocr, get_bounded_ocr

def apply_ocr(path, image):
    """
    ARGUMENTS:
        path: if ocr data already exists for the given path
        image: Image object that ocr needs to be applied on
    RETURNS:
        bboxes: entire ocr data of the Image
    """

    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        _w, _h = image.size
        _r = 2500 / _w
        image = image.resize((2500, int(_r * _h)))

        print("OCR start")
        ocr = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        print("OCR end")

        bboxes = []
        for i in range(len(ocr["conf"])):
            if ocr["level"][i] > 4 and ocr["text"][i].strip() != "" and ocr['conf'][i] > 50:
                bboxes.append(
                    [
                        len(ocr["text"][i]),
                        ocr["text"][i],
                        int(ocr["left"][i] / _r),
                        int(ocr["top"][i] / _r),
                        int(ocr["left"][i] / _r) + int(ocr["width"][i] / _r),
                        int(ocr["top"][i] / _r) + int(ocr["height"][i] / _r),
                    ]
                )

        bboxes = sorted(
            bboxes, key=lambda box: (box[4] - box[2]) * (box[5] - box[3]), reverse=True
        )
        threshold = np.average(
            [
                (box[4] - box[2]) * (box[5] - box[3])
                for box in bboxes[len(bboxes) // 20 : -len(bboxes) // 4]
            ]
        )
        bboxes = [
            box
            for box in bboxes
            if (box[4] - box[2]) * (box[5] - box[3]) < threshold * 30
        ]

        with open(path, "wb") as f:
            pickle.dump(bboxes, f)

        return bboxes

def remove_background(image):
    image2 = np.copy(image)
    kernel = np.ones((1, 5), np.uint8)
    lines1 = np.copy(image)
    lines1 = cv2.dilate(lines1, kernel, iterations=17)
    lines1 = cv2.erode(lines1, kernel, iterations=17)

    kernel = np.ones((5, 1), np.uint8)
    lines2 = np.copy(image)
    lines2 = cv2.dilate(lines2, kernel, iterations=17)
    lines2 = cv2.erode(lines2, kernel, iterations=17)

    lines2 = np.uint8(np.clip(np.int16(lines2) - np.int16(lines1) + 255, 0, 255))
    lines = np.uint8(np.clip((255 - np.int16(lines1)) + (255 - np.int16(lines2)), 0, 255))

    bg_removed = np.uint8(np.clip(np.int16(image2) + np.int16(lines), 0, 255))

    return bg_removed



def extract_lines(image, x=17, y=17):
    # image = cv2.erode(image.copy(), np.ones((3,3), np.uint8), iterations=1)

    kernel = np.ones((1, 5), np.uint8)
    lines1 = np.copy(image)
    lines1 = cv2.dilate(lines1, kernel, iterations=x)
    lines1 = cv2.erode(lines1, kernel, iterations=x)

    kernel = np.ones((5, 1), np.uint8)
    lines2 = np.copy(image)
    lines2 = cv2.dilate(lines2, kernel, iterations=y)
    lines2 = cv2.erode(lines2, kernel, iterations=y)

    lines2, lines1 = np.uint8(np.clip(np.int16(lines2) - np.int16(lines1) + 255, 0, 255)), \
                     np.uint8(np.clip(np.int16(lines1) - np.int16(lines2) + 255, 0, 255))
    lines = np.uint8(np.clip((255 - np.int16(lines1)) + (255 - np.int16(lines2)), 0, 255))
    return lines



def process_files(image_dir, xml_dir, ocr_dir, out_dir):
    """
    ARGUMENTS:
        image_dir: directory of the document image file
        xml_dir: directory of the xml file
        ocr_dir: directory of the ocr file
        out_dir: the output directory for saving data
        
    RETURNS:
        returns no data, saves the processed data to the provided output directory.
    """
    files = [
        file.split("/")[-1].rsplit(".", 1)[0]
        for file in glob.glob(os.path.join(xml_dir, "*.xml"))
    ]
    files.sort()

    for ii, file in enumerate(files):
        try:
            image_file = os.path.join(image_dir, file + ".png")
            xml_file = os.path.join(xml_dir, file + ".xml")
            ocr_file = os.path.join(ocr_dir, file + ".pkl")

            img = cv2.imread(image_file)

            # tmp = img.copy()
            # tmp = cv2.GaussianBlur(tmp,(5,5),0)
            # tmp = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            # lines = extract_lines(tmp)
            # lines = cv2.dilate(lines, np.ones((3,3), np.uint8), iterations=1)
            # ocr_img = np.uint8(np.clip(np.int16(img) + np.int16(lines), 0, 255))

            ocr = apply_ocr(ocr_file, Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))

            if (
                os.path.exists(image_file)
                and os.path.exists(xml_file)
                and os.path.exists(ocr_file)
            ):
                print("[", ii, "/", len(files), "]", "Processing: ", file)
                doc = Document(xml_file)

                # cv2.imshow("img", cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
                # cv2.imshow("ocr", cv2.resize(ocr_img, (img.shape[1] // 2, img.shape[0] // 2)))
                # cv2.waitKey(0)
                # exit(0)

                for i, obj in enumerate(doc.tables):
                    table_name = file + "_" + str(i)

                    img_crop = img[obj.y1: obj.y2, obj.x1: obj.x2]
                    table_ocr = translate_ocr(
                        get_bounded_ocr(ocr, (obj.x1, obj.y1), (obj.x2, obj.y2)), 
                        (-obj.x1, -obj.y1)
                    )
                    obj.move(-obj.x1, -obj.y1)

                    cv2.imwrite(
                        os.path.join(out_dir, "images", table_name + ".png"), img_crop
                    )

                    dummy_doc = Document()
                    dummy_doc.tables.append(obj)
                    dummy_doc.input_file = table_name + '.png'
                    dummy_doc.write_to(os.path.join(out_dir, "gt", table_name + '.xml'))

                    with open(
                        os.path.join(out_dir, "ocr", table_name + ".pkl"), "wb"
                    ) as f:
                        pickle.dump(table_ocr, f)
        except Exception as e:
            print(file)
            print(e)

if __name__ == "__main__":
    _parser = argparse.ArgumentParser()
    _parser.add_argument(
        "-img",
        "--image_dir",
        type=str,
        help="Directory containing document-level images",
        default="/home/umar_visionx/Documents/Asad/data/test/testv2/images_doc",
        required=True,
    )

    _parser.add_argument(
        "-xml",
        "--xml_dir",
        type=str,
        help="Directory containing document-level xmls",
        default="/home/umar_visionx/Documents/Asad/data/test/testv2/gt_doc",
        required=True,
    )

    _parser.add_argument(
        "-ocr",
        "--ocr_dir",
        type=str,
        help="Directory containing document-level ocr files. (If an OCR file is not found, it will be generated and saved in this directory for future use)",
        default="/home/umar_visionx/Documents/Asad/data/test/testv2/ocr_doc",
        required=True,
    )

    _parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        help="Path of output directory for generated data",
        default="/home/umar_visionx/Documents/Asad/data/test/testv2/ocr_tab",
        required=True,
    )

    args = _parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "gt"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "ocr"), exist_ok=True)

    process_files(args.image_dir, args.xml_dir, args.ocr_dir, args.out_dir)
