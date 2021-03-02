training_name=${1}
train_dir=./trainings
out_dir=./outputs

python infer.py -img ../data/cropped/icdar/test/images -m ${train_dir}/${training_name}/best_model.pth -o ${out_dir}/${training_name}/
