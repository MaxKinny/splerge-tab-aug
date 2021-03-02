import os
import csv
import glob
import shutil
import numpy as np
import pandas as pd

if __name__ == "__main__":
    eval_dir = 'new_evals'
    out_dir = 'new_evals/combined'

    eval_names = list(map(os.path.basename, glob.glob(os.path.join(eval_dir, "1", "*"))))

    keys = ['row', 'col', 'cell']
    # keys = ['row', 'col']
    metric_names = [
                    'correct',
                    'partial',
                    'over-seg',
                    'under-seg',
                    'missed',
                    'false-positives',
                    'num-pred',
                    'num-gt'
                ]

    correct_vals = {}
    for eval_name in eval_names:
        dfs = []
        for i in range(1, 4):
            df = pd.read_csv(os.path.join(eval_dir, str(i), eval_name, "evaluation.csv"))
            df = df[keys]
            dfs.append(df)

        correct_vals[eval_name] = []
        with open(os.path.join(out_dir, eval_name + '.csv'), "w") as f:
            csv_writer = csv.writer(f, delimiter=",")

            csv_writer.writerow([""] + [key+x for key in keys for x in ['_mean', '_std']])

            means = {}
            stds = {}
            for key in dfs[0].columns:
                series = np.array([df[key].to_numpy() for df in dfs])
                means[key] = series.mean(axis=0)
                stds[key] = series.std(axis=0)

            for i, name in enumerate(metric_names):
                tmp = []
                for key in means.keys():
                    tmp.append(means[key][i])
                    tmp.append(stds[key][i])
                csv_writer.writerow([name] + tmp)



        # for key in dfs[0].columns:
        #   series = np.array([df[key].to_numpy() for df in dfs])
        #   mean = series.mean(axis=0)
        #   std = series.std(axis=0)
