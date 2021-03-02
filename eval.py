import os
import csv
import glob
import string
import pickle
import random
import argparse

import cv2
import numpy as np
from truthpy import Document, Rect

class Evaluator:
    def __init__(self, images_dir, ocr_dir, gt_dir, pred_dir, output_dir):
        self.images_dir = images_dir
        self.ocr_dir = ocr_dir
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir

        self.output_dir = output_dir
        self.output_images_dir = os.path.join(output_dir, "visualizations")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_images_dir, exist_ok=True)

        self.filenames = \
            list(
                map(
                    lambda x: os.path.basename(x).rsplit('.', 1)[0], 
                    glob.glob(os.path.join(gt_dir, "*.xml"))
                )
            )

        types = ['row', 'col', 'cell']
        self.metric_names = [
                    'correct',
                    'partial',
                    'over-seg',
                    'under-seg',
                    'missed',
                    'false-positives',
                    'num-pred',
                    'num-gt'
                ]

        self.metrics = {}
        for t in types:
            self.metrics[t] = {}
            for name in self.metric_names:
                self.metrics[t][name] = 0

        self.colors = [0] * 24000 # self.generate_random_colors()
        for i in range(100):
            self.colors[i*240:(i+1)*240:2] = list(range(10 + i*256, 130 + i*256))[::-1]
            self.colors[1+i*240:(i+1)*240:2] = list(range(130 + i*256, 250 + i*256))[::-1]
        # random.shuffle(self.colors)

    def evaluate(self):

        for filename in self.filenames:
            print(filename)
            image       = cv2.imread(os.path.join(self.images_dir, filename + '.png'))
            assert image.shape[2] == 3
            table_gt    = Document(os.path.join(self.gt_dir, filename + '.xml')).tables[0]
            table_pred  = Document(os.path.join(self.pred_dir, filename + '.xml')).tables[0]
            with open(os.path.join(self.ocr_dir, filename + '.pkl'), "rb") as f:
                ocr = pickle.load(f)

            rects_gt    = self.get_rects_from_tables(table_gt)
            rects_pred  = self.get_rects_from_tables(table_pred)

            masks = {}
            masks['row']    = self.get_ocr_mask(image, table_gt, ocr, ignore_rowspan=True)
            masks['col']    = self.get_ocr_mask(image, table_gt, ocr, ignore_colspan=True)
            masks['cell']   = masks['row'] & masks['col']

            for eval_name in self.metrics.keys():

                gt_color_encoded    = np.zeros(image.shape[:2], dtype=np.int16)
                pred_color_encoded  = np.zeros(image.shape[:2], dtype=np.int16)

                for i, rect in enumerate(rects_gt[eval_name]):
                    cv2.rectangle(gt_color_encoded, (rect.x1, rect.y1), (rect.x2, rect.y2), self.colors[i], -1)

                for i, rect in enumerate(rects_pred[eval_name]):
                    cv2.rectangle(pred_color_encoded, (rect.x1, rect.y1), (rect.x2, rect.y2), self.colors[i], -1)

                gt_color_encoded    = gt_color_encoded & masks[eval_name]
                pred_color_encoded  = pred_color_encoded & masks[eval_name]

                pred_colors = self.colors[:len(rects_pred[eval_name])]
                gt_colors   = self.colors[:len(rects_gt[eval_name])]

                results = self.evaluate_color_encodings(gt_color_encoded, pred_color_encoded, gt_colors, pred_colors)

                pred_color_encoded = (pred_color_encoded % 256).astype(np.uint8)
                gt_color_encoded = (gt_color_encoded % 256).astype(np.uint8)

                pred_color_encoded  = cv2.cvtColor(pred_color_encoded, cv2.COLOR_GRAY2BGR)
                gt_color_encoded    = cv2.cvtColor(gt_color_encoded, cv2.COLOR_GRAY2BGR)

                for i, name in enumerate(self.metric_names):
                    self.metrics[eval_name][name] += results[name]
                    cv2.putText(
                        pred_color_encoded, "{} : {}".format(name, results[name]), (10, i*40 + 40), 
                        cv2.FONT_HERSHEY_SIMPLEX,  
                        0.7, (0, 255, 0), 2, cv2.LINE_AA
                    )

                cv2.imwrite(os.path.join(self.output_images_dir, filename + '_{}_pred.png'.format(eval_name)), pred_color_encoded)
                cv2.imwrite(os.path.join(self.output_images_dir, filename + '_{}_gt.png'.format(eval_name)), gt_color_encoded)
            
            cv2.imwrite(os.path.join(self.output_images_dir, filename + '.png'), image)

        with open(os.path.join(self.output_dir, "evaluation.csv"), "w") as f:
            csv_writer = csv.writer(f, delimiter=",")

            csv_writer.writerow([""] + [key for key in self.metrics])
            for name in self.metric_names:
                if name in ["num-pred", 'num-gt']:
                    csv_writer.writerow([name] + [self.metrics[key][name] for key in self.metrics])
                else:
                    csv_writer.writerow([name] + [round(100 * self.metrics[key][name] / self.metrics[key]["num-gt"], 2) for key in self.metrics])
            # csv_writer.writerow(
            #     ["False Negatives"]
            #     + [self.metrics[key]["missed"] for key in self.metrics]
            # )
            # csv_writer.writerow(
            #     ["False Positives"]
            #     + [self.metrics[key]["incorrect"] for key in self.metrics]
            # )

            # csv_writer.writerow(
            #     ["Precision"]
            #     + [
            #         str(
            #             round(
            #                 self.metrics[key]["correct"]
            #                 * 100
            #                 / max(
            #                     1,
            #                     self.metrics[key]["correct"]
            #                     + self.metrics[key]["incorrect"],
            #                 ),
            #                 2,
            #             )
            #         )
            #         + "%"
            #         for key in self.metrics
            #     ]
            # )
            # csv_writer.writerow(
            #     ["Recall"]
            #     + [
            #         str(
            #             round(
            #                 self.metrics[key]["correct"]
            #                 * 100
            #                 / max(
            #                     1,
            #                     self.metrics[key]["correct"]
            #                     + self.metrics[key]["missed"],
            #                 ),
            #                 2,
            #             )
            #         )
            #         + "%"
            #         for key in self.metrics
            #     ]
            # )
    def evaluate_color_encodings(self, gt_img, pred_img, gt_colors, pred_colors, T=0.1):
        c_matrix = np.zeros((len(gt_colors), len(pred_colors)))

        # print((len(gt_colors), len(pred_colors)))

        gt_masks = []
        pred_masks = []
        for i in range(c_matrix.shape[0]):
            gt_masks.append((gt_img == gt_colors[i]))
        for j in range(c_matrix.shape[1]):
            pred_masks.append((pred_img == pred_colors[j]))

        for j in range(c_matrix.shape[1]):
            if np.count_nonzero(pred_masks[j]) > 0:
                for i in range(c_matrix.shape[0]):
                    c_matrix[i, j] += np.count_nonzero(np.logical_and(gt_masks[i], pred_masks[j]))
                # print("majsdhgasd")
        
        gt_overlaps   = c_matrix / np.maximum(1, c_matrix.sum(axis=1)[:, None])
        pred_overlaps = c_matrix / np.maximum(1, c_matrix.sum(axis=0)[None, :])

        _correct = 0
        _partial = 0
        for i in range(c_matrix.shape[0]):
            pred = np.argmax(c_matrix[i, :])
            if np.count_nonzero(pred_overlaps[:, pred] > T) == 1:
                if np.count_nonzero(gt_overlaps[i, :] > 1 - T) == 1:
                    _correct += 1
                elif np.count_nonzero(gt_overlaps[i, :] > T) == 1:
                    _partial += 1

        _over_segmentations     = np.count_nonzero(((1 - T > gt_overlaps) & (gt_overlaps > T)).sum(axis=1) > 1)
        _under_segmentations    = np.count_nonzero(((1 - T > pred_overlaps) & (pred_overlaps > T)).sum(axis=0) > 1)
        _missed                 = np.count_nonzero((np.count_nonzero(gt_overlaps   > T, axis=1) == 0) & (c_matrix.sum(axis=1) > 0))
        _false_positives        = np.count_nonzero((np.count_nonzero(pred_overlaps > T, axis=0) == 0) & (c_matrix.sum(axis=0) > 0))

        return {
            "correct": _correct,
            "partial": _partial,
            "over-seg": _over_segmentations,
            "under-seg": _under_segmentations,
            "missed": _missed,
            "false-positives": _false_positives,
            "num-pred": np.count_nonzero(c_matrix.sum(axis=0) > 0),
            "num-gt": np.count_nonzero(c_matrix.sum(axis=1) > 0)
        }

    def get_ocr_mask(self, image, table, ocr, ignore_rowspan=False, ignore_colspan=False):
        ocr_mask = np.zeros(image.shape[:2], dtype=np.int16)
        for word in ocr:
            cv2.rectangle(ocr_mask, (word[2], word[3]), (word[4], word[5]), 0xFFFF, -1)

        for row in table.gtCells:
            for cell in row:
                if (
                    (ignore_colspan and cell.startCol != cell.endCol)
                    or 
                    (ignore_rowspan and cell.startRow != cell.endRow)
                ):
                    cv2.rectangle(ocr_mask, (cell.x1, cell.y1), (cell.x2, cell.y2), 0, -1)

        return ocr_mask            

    def get_rects_from_tables(self, table):
        rects = {'row': [], 'col': [], 'cell': []}

        row_sep = [table.y1] + [r.y1 for r in table.gtRows] + [table.y2]
        for row1, row2 in zip(row_sep[:-1], row_sep[1:]):
            rects['row'].append(Rect(table.x1, row1, table.x2, row2))

        col_sep = [table.x1] + [r.x1 for r in table.gtCols] + [table.x2]
        for col1, col2 in zip(col_sep[:-1], col_sep[1:]):
            rects['col'].append(Rect(col1, table.y1, col2, table.y2))

        rects['cell'] = [cell for row in table.gtCells for cell in row if not cell.dontCare]

        return rects

    def generate_random_colors(self):
        colors = []
        for r in range(30, 230, 20):
            for g in range(30, 230, 20):
                for b in range(30, 230, 20):
                    if b == g and g == r:
                        continue
                    colors.append((b, g, r))

        random.shuffle(colors)
        return colors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images_dir", type=str, required=True,
                        help="path to directory containing table-level images.")

    parser.add_argument("-xml", "--xml_dir", type=str, required=True,
                        help="path to directory containing table-level ground-truth XML files.")

    parser.add_argument("-o", "--ocr_dir", type=str, required=True,
                        help="path to directory containing table-level ocr.")

    parser.add_argument("-p", "--pred_dir", type=str, required=True,
                        help="path to directory containing table-level prediction XML files.")

    parser.add_argument("-e", "--eval_out", type=str, required=True,
                        help="path of directory in which to write the evaluation results.")

    args = parser.parse_args()

    os.makedirs(args.eval_out, exist_ok=True)
    os.makedirs(os.path.join(args.eval_out, "visualizations"), exist_ok=True)

    Evaluator(args.images_dir, args.ocr_dir, args.xml_dir, args.pred_dir, args.eval_out).evaluate()