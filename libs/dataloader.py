import os
import pickle
import string

import torch
import numpy as np
import cv2

from termcolor import cprint

from libs.utils import resize_image
from libs.utils import normalize_numpy_image

from truthpy import Document
from augmentation.augmentor import augment_table

class SplitTableDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        train_images_path,
        train_labels_path,
        train_ocr_path,
        transforms=None,
        fix_resize=False,
        augment=False
    ):

        self.fix_resize = fix_resize
        self.root = root
        self.transforms = transforms
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        self.train_ocr_path = train_ocr_path

        self.augment = augment

        # cprint(self.root, "yellow")
        # cprint(self.train_images_path, "yellow")
        # cprint(self.train_labels_path, "yellow")

        self.filenames = list(
            sorted(os.listdir(os.path.join(self.root, self.train_images_path)))
        )
        self.filenames = list(map(lambda name: os.path.basename(name).rsplit('.', 1)[0], self.filenames))

    def read_record(self, idx):
        filename = self.filenames[idx]
        image_file = os.path.join(self.train_images_path, filename + ".png")
        xml_file = os.path.join(self.train_labels_path, filename + ".xml")
        ocr_file = os.path.join(self.train_ocr_path, filename + ".pkl")

        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

        with open(ocr_file, "rb") as f:
            ocr = pickle.load(f)
        doc = Document(xml_file)
        assert len(doc.tables) == 1
        table = doc.tables[0]

        if self.augment is True:
            return_val = augment_table(table, img.copy(), ocr.copy())
            if return_val is not False:
                table, img, ocr = return_val
            else:
                table = Document(xml_file).tables[0]


        ocr_mask = np.zeros_like(img)
        for word in ocr:
            txt = word[1].translate(str.maketrans("", "", string.punctuation))
            if len(txt.strip()) > 0:
                cv2.rectangle(ocr_mask, (word[2], word[3]), (word[4], word[5]), 255, -1)
        ocr_mask_row = ocr_mask.copy()
        # cv2.imshow("mask", ocr_mask)

        columns = [1] + [col.x1 for col in table.gtCols] + [img.shape[1] - 1]
        rows = [1] + [row.y1 for row in table.gtRows] + [img.shape[0] - 1]

        for row in table.gtCells:
            for cell in row:
                x0, y0, x1, y1 = tuple(cell)
                if cell.startRow != cell.endRow:
                    cv2.rectangle(ocr_mask_row, (x0, y0), (x1, y1), 0, -1)
                if cell.startCol != cell.endCol:
                    cv2.rectangle(ocr_mask, (x0, y0), (x1, y1), 0, -1)

        col_gt_mask = np.zeros_like(img[0, :])
        row_gt_mask = np.zeros_like(img[:, 0])

        non_zero_rows = np.append(
            np.where(np.count_nonzero(ocr_mask_row, axis=1) != 0)[0],
            [-1, img.shape[0]],
        )
        non_zero_cols = np.append(
            np.where(np.count_nonzero(ocr_mask, axis=0) != 0)[0],
            [-1, img.shape[1]],
        )
        zero_rows = np.where(np.count_nonzero(ocr_mask_row, axis=1) == 0)[0]
        zero_cols = np.where(np.count_nonzero(ocr_mask, axis=0) == 0)[0]

        for col in columns:
            if col == 0 or col == img.shape[1]:
                continue
            diff = non_zero_cols - col
            left = min(-diff[diff < 0]) - 1
            right = min(diff[diff > 0])

            # Re-align the seperators passing through an ocr bounding box
            try:
                if left == 0 and right == 1:
                    if col == 1 or col == img.shape[1] - 1:
                        continue
                    diff_zeros = zero_cols - col
                    left_align = min(-diff_zeros[diff_zeros < 0])
                    right_align = min(diff_zeros[diff_zeros > 0])

                    if min(left_align, right_align) < 20:
                        if left_align < right_align:
                            col -= left_align
                        else:
                            col += right_align

                        diff = non_zero_cols - col
                        left = min(-diff[diff < 0]) - 1
                        right = min(diff[diff > 0])
            except Exception as e:
                pass

            col_gt_mask[col - left : col + right] = 255

        for row in rows:
            if row == 0 or row == img.shape[0]:
                continue
            diff = non_zero_rows - row
            above = min(-diff[diff < 0]) - 1
            below = min(diff[diff > 0])

            # Re-align the seperators passing through an ocr bounding box
            try:
                if above == 0 and below == 1:
                    if row == 1 or row == img.shape[0] - 1:
                        continue
                    diff_zeros = zero_rows - row
                    above_align = min(-diff_zeros[diff_zeros < 0])
                    below_align = min(diff_zeros[diff_zeros > 0])

                    if min(above_align, below_align) < 20:
                        if above_align < below_align:
                            row -= above_align
                        else:
                            row += below_align

                        diff = non_zero_rows - row
                        above = min(-diff[diff < 0]) - 1
                        below = min(diff[diff > 0])
            except Exception as e:
                pass

            row_gt_mask[row - above : row + below] = 255
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32), row_gt_mask, col_gt_mask

    def __getitem__(self, idx):
        image, row_label, col_label = self.read_record(idx)

        if image.ndim == 2:
            # reshape (H, W) -> (1, H, W)
            image = image[np.newaxis]
        else:
            # transpose (H, W, C) -> (C, H, W)
            image = image.transpose((2, 0, 1))

        C, H, W = image.shape
        image = resize_image(image, fix_resize=self.fix_resize)

        # image_write = image.copy()
        # image_write = (image_write.transpose((1, 2, 0))*255).astype(np.uint8)
        # cv2.imwrite("debug/{}_image.png".format(self.filenames[idx]), image_write)

        image = normalize_numpy_image(image)

        image = image.numpy()

        _, o_H, o_W = image.shape
        scale = o_H / H

        row_label = cv2.resize(row_label[np.newaxis, :], (o_H, 1), interpolation=cv2.INTER_NEAREST)
        col_label = cv2.resize(col_label[np.newaxis, :], (o_W, 1), interpolation=cv2.INTER_NEAREST)

        # image_write[row_label[0] == 255, :, :] = [255, 0, 255]
        # image_write[:, col_label[0] == 255, :] = [255, 0, 255]
        # cv2.imwrite("debug/{}_labels.png".format(self.filenames[idx]), image_write)

        row_label[row_label > 0] = 1
        col_label[col_label > 0] = 1

        row_label = torch.tensor(row_label[0])
        col_label = torch.tensor(col_label[0])

        target = [row_label, col_label]

        image = image.transpose((1, 2, 0))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target, self.filenames[idx], W, H

    def __len__(self):
        return len(self.filenames)


class MergeTableDataset(torch.utils.data.Dataset):
    def __init__(self, root, train_features_path, train_labels_path, transforms=None):
        self.root = root
        self.train_features_path = train_features_path
        self.train_labels_path = train_labels_path
        self.transforms = transforms

        self.feature_paths_list = list(
            sorted(os.listdir(os.path.join(self.root, self.train_features_path)))
        )

    def __getitem__(self, idx):
        feature_path = os.path.join(
            self.root, self.train_features_path, self.feature_paths_list[idx]
        )
        file_name = self.feature_paths_list[idx][:-4]
        target_path = os.path.join(self.root, self.train_labels_path, file_name)

        with open(feature_path, "rb") as f:
            input_feature = pickle.load(feature_path)

        with open(target_path, "rb") as f:
            target = pickle.load(target_path)

        # if self.transforms is not None:
        #     image, target = self.transforms(image, target)

        return input_feature, target, feature_path

    def __len__(self):
        return len(self.img_paths)

