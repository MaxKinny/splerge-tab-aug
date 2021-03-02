import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

labels = ['25%', '50%', '75%', '100%']

############ ROW ###########
mean_org = [94.59, 94.52, 95.37, 96.44]
std_org = [0.27, 1.4, 1.19, 1.13]

mean_aug = [96.37, 97.15, 97.79, 97.86]
std_aug = [0.7, 0.44, 0.66, 0.35]

# mean_cls = [96.15, 95.02, 94.73, 97.86]
# std_cls = [0.52, 1.41, 3.38, 0.8]
# mean_cls = [96.15, 95.02, 95.91, 97.86]
# std_cls = [0.52, 1.41, 2.67, 0.8]
mean_cls = [96.15, 95.02, 95.97, 97.86]
std_cls = [0.52, 1.41, 1.65, 0.8]

mean_org = np.array(mean_org)
std_org = np.array(std_org)
mean_aug = np.array(mean_aug)
std_aug = np.array(std_aug)
mean_cls = np.array(mean_cls)
std_cls = np.array(std_cls)

def save_plot(name):
	plt.xlabel("Training Set Used")
	plt.ylabel("Correct Detections (%)")
	plt.plot(labels, mean_aug, 'o-b', label="TabAug")
	plt.plot(labels, mean_org, '^--r', label="Non-Augmented")
	plt.plot(labels, mean_cls, 'D:g', label="Standard")
	plt.fill_between(labels, mean_aug-std_aug, mean_aug+std_aug, facecolor='blue', alpha=0.2)
	plt.fill_between(labels, mean_org-std_org, mean_org+std_org, facecolor='red', alpha=0.2)
	plt.fill_between(labels, mean_cls-std_cls, mean_cls+std_cls, facecolor='green', alpha=0.2)
	plt.legend()
	plt.savefig(name, dpi=600)
	plt.clf()
	img = cv2.imread(name)
	img = img[150:-30, 90:-180]
	cv2.imwrite(name, img)

save_plot("ROW.png")

############ COLUMN ###########
mean_org = [80.46, 89.96, 89.79, 92.12]
std_org = [4.08, 1.34, 2.88, 1.11]

mean_aug = [84.05, 94.8, 94.98, 94.44]
std_aug = [2.65, 1.27, 1.26, 0.25]

mean_cls = [78.67, 83.34, 84.41, 86.38]
std_cls = [2.92, 1.91, 3.8, 1.54]
# mean_cls = [78.67, 83.34, 85.89, 86.38]
# std_cls = [2.92, 1.91, 4.17, 1.54]

mean_org = np.array(mean_org)
std_org = np.array(std_org)
mean_aug = np.array(mean_aug)
std_aug = np.array(std_aug)
mean_cls = np.array(mean_cls)
std_cls = np.array(std_cls)

save_plot("COLUMN.png")
# plt.plot(labels, mean_aug, 'o-b')
# plt.plot(labels, mean_org, 'o--r')
# plt.plot(labels, mean_cls, 'o:g')
# plt.fill_between(labels, mean_aug-std_aug, mean_aug+std_aug, facecolor='blue', alpha=0.3)
# plt.fill_between(labels, mean_org-std_org, mean_org+std_org, facecolor='red', alpha=0.3)
# plt.fill_between(labels, mean_cls-std_cls, mean_cls+std_cls, facecolor='green', alpha=0.3)
# plt.savefig('COLUMN.png')
# plt.clf()

############ CELL ###########
mean_org = [76.07, 82.92, 84.17, 92.16]
std_org = [2.56, 2.17, 3.51, 3.84]

mean_aug = [81.84, 93.73, 95.14, 96.11]
std_aug = [3.68, 2.81, 3.01, 1.61]

mean_cls = [73.76, 68.99, 76.45, 82.12]
std_cls = [2.08, 3.5, 2.55, 6.76]
# mean_cls = [73.76, 68.99, 80.06, 82.12]
# std_cls = [2.08, 3.5, 6.64, 6.76]

mean_org = np.array(mean_org)
std_org = np.array(std_org)
mean_aug = np.array(mean_aug)
std_aug = np.array(std_aug)
mean_cls = np.array(mean_cls)
std_cls = np.array(std_cls)

save_plot("CELL.png")
# plt.plot(labels, mean_aug, 'o-b')
# plt.plot(labels, mean_org, 'o--r')
# plt.plot(labels, mean_cls, 'o:g')
# plt.fill_between(labels, mean_aug-std_aug, mean_aug+std_aug, facecolor='blue', alpha=0.3)
# plt.fill_between(labels, mean_org-std_org, mean_org+std_org, facecolor='red', alpha=0.3)
# plt.fill_between(labels, mean_cls-std_cls, mean_cls+std_cls, facecolor='green', alpha=0.3)
# plt.savefig('CELL.png')
# plt.clf()