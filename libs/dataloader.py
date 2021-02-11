import os
import pickle
import string
import random

import cv2
import torch
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF

from termcolor import cprint

from libs.utils import resize_image
from libs.utils import normalize_numpy_image

from truthpy import Document
from augmentation.augmentor import Augmentor, apply_action

def generate_gauss(center, shape=(5, 5)):
    gauss = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            gauss[i, j] = 1 / (2 ** max(0, abs((i - center[0])) + abs((j - center[1])) - 1))
    return gauss

class SplitTableDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        fix_resize=False,
        augment=False,
        classical_augment=False,
    ):

        self.fix_resize = fix_resize
        self.train_images_path = os.path.join(root, "images")
        self.train_labels_path = os.path.join(root, "gt")
        self.train_ocr_path    = os.path.join(root, "ocr")

        self.augment = augment

        # cprint(self.root, "yellow")
        # cprint(self.train_images_path, "yellow")
        # cprint(self.train_labels_path, "yellow")

        self.filenames = list(
            sorted(os.listdir(self.train_images_path))
        )
        self.filenames = list(map(lambda name: os.path.basename(name).rsplit('.', 1)[0], self.filenames))

        self.col_steps = [0, 4, 6, 9]
        self.row_steps = [0, 4, 6, 9, 13]
        if self.augment:
            print("Reading:  Augmentation Data...")
            self.distribution = self.read_distributions()
            self.nodes, self.probs = self.read_nodes()
            print("Complete: Augmentation Data.")

        self.classical_augment = classical_augment
        if self.classical_augment:
            self.classical_transform = transforms.Compose([
                transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.7,2.5), saturation=(0,2), hue=0.5),

            ])

    def read_distributions(self):
        with open("distributions/icdar_metadata.pkl", "rb") as f:
            distributions = pickle.load(f)

        rc_categorized = np.zeros((len(self.row_steps), len(self.col_steps)))
        rc_distribution = distributions[0]

        for i, (row_start, row_end) in enumerate(zip(self.row_steps[:-1], self.row_steps[1:])):
            for j, (col_start, col_end) in enumerate(zip(self.col_steps[:-1], self.col_steps[1:])):
                rc_categorized[i, j] = rc_distribution[row_start: row_end, col_start:col_end].sum()
            rc_categorized[i, -1] = rc_distribution[row_start: row_end, self.col_steps[-1]:].sum()

        for j, (col_start, col_end) in enumerate(zip(self.col_steps[:-1], self.col_steps[1:])):
            rc_categorized[-1, j] = rc_distribution[self.row_steps[-1]:, col_start:col_end].sum()

        rc_categorized[-1, -1] = rc_distribution[self.row_steps[-1]:, self.col_steps[-1]:].sum()
        rc_categorized += 0.01

        return rc_categorized

    def read_nodes(self):
        with open("distributions/icdar_nodes_3.pkl", "rb") as f:
            file_to_nodes = pickle.load(f)

        file_to_probs = {}
        for filename in file_to_nodes.keys():
            categorized = [[[] for i in range(len(self.col_steps))] for j in range(len(self.row_steps))]

            gauss = None
            def find_bin(seperators, val):
                for i, sep in enumerate(seperators):
                    if val < sep:
                        return i - 1
                return len(seperators) - 1

            for idx, node in enumerate(file_to_nodes[filename]):
                r_idx = find_bin(self.row_steps, node[0]['h'])
                c_idx = find_bin(self.col_steps, node[0]['w'])

                categorized[r_idx][c_idx].append(node)

            doc = Document(os.path.join(self.train_labels_path, filename + ".xml"))
            assert len(doc.tables) == 1
            table = doc.tables[0]
    
            r_idx = find_bin(self.row_steps, len(table.gtCells))
            c_idx = find_bin(self.col_steps, len(table.gtCells[0]))
            gauss = generate_gauss((r_idx, c_idx), shape=(len(self.row_steps), len(self.col_steps)))

            freqs = np.array(list(map(lambda x: list(map(len, x)), categorized)))

            probs = gauss * self.distribution * freqs
            probs = probs / probs.sum()
            file_to_nodes[filename] = categorized
            file_to_probs[filename] = probs
        return file_to_nodes, file_to_probs

    def apply_augmentation(self, filename, table, img, ocr):
        augmentor = Augmentor(table, img, ocr)

        nodes = self.nodes[filename]
        probs = self.probs[filename]

        # Select which size to choose from.
        i = np.random.choice(np.arange(probs.size), p=probs.ravel())
        indices = np.unravel_index(i, probs.shape)

        chosen_node = random.choice(nodes[indices[0]][indices[1]])

        for action in chosen_node[1]:
            return_val = apply_action(augmentor, action)
            assert return_val

        table, img, ocr = augmentor.t, augmentor.image, augmentor.ocr
        assert len(table.gtCells)==chosen_node[0]['h']
        assert len(table.gtCells[0])==chosen_node[0]['w']
        return table, img, ocr

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
            # cv2.imshow("before aug", img)
            table, img, ocr = self.apply_augmentation(filename, table, img.copy(), ocr.copy())
            # cv2.imshow("after  aug", img)
            # cv2.waitKey(0)

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
            image = image[:, :, np.newaxis]

        # if self.classical_augment and random.random() > 0.8:
        #     # Cropping
        #     if random.random() > 0.5:
        #         h, w = image.shape[1:]
        #         crop_w = random.randint(int(w * 0.6), w)
        #         x = random.randint(0, (w - crop_w))

        #         crop_h = random.randint(int(h * 0.6), h)
        #         y = random.randint(0, (h - crop_h))

        #         image = image[:, y: y+crop_h, x: x + crop_w]
        #         row_label = row_label[y: y+crop_h]
        #         col_label = col_label[x: x+crop_w]


        H, W, C = image.shape
        image = resize_image(image, fix_resize=self.fix_resize)

        # image_write = image.copy()
        # image_write = (image_write.transpose((1, 2, 0))*255).astype(np.uint8)
        # cv2.imwrite("debug/{}_image.png".format(self.filenames[idx]), image_write)

        o_H, o_W, _ = image.shape
        scale = o_H / H

        row_label = cv2.resize(row_label[np.newaxis, :], (o_H, 1), interpolation=cv2.INTER_NEAREST)
        col_label = cv2.resize(col_label[np.newaxis, :], (o_W, 1), interpolation=cv2.INTER_NEAREST)

        # image_write[row_label[0] == 255, :, :] = [255, 0, 255]
        # image_write[:, col_label[0] == 255, :] = [255, 0, 255]
        # cv2.imshow("labels.png", image_write)
        # cv2.waitKey(0)

        row_label[row_label > 0] = 1
        col_label[col_label > 0] = 1

        row_label = torch.tensor(row_label[0])
        col_label = torch.tensor(col_label[0])

        target = [row_label, col_label]

        image = image.transpose((2, 0, 1))
        image = normalize_numpy_image(image)

        return image, target, self.filenames[idx], W, H

    def __len__(self):
        return len(self.filenames)


class MergeTableDataset(torch.utils.data.Dataset):
    def __init__(self, root, train_features_path, train_labels_path, transform=None):
        self.root = root
        self.train_features_path = train_features_path
        self.train_labels_path = train_labels_path
        self.transforms = transform

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

