suffix=/icdar
augment=${1}
classical_augment=${2}

root_data_path=../data/cropped
data_path=${root_data_path}/${suffix}/train${3}
val_path=${root_data_path}/${suffix}/val
out_path=$suffix

if [ "$augment" = true ]; then
  out_path=${out_path}-augmented
  args="--augment_tables"
fi

if [ "$classical_augment" = true ]; then
  out_path=${out_path}-classical
  args="${args} --classical_augment"
fi

out_path=./trainings/${out_path}-${3}
echo $args
# rm -r ${out_path}
python train.py --train_dir ${data_path} --val_dir ${val_path} -o ${out_path} -e ${4} ${args} --dr 0.8 --val_every ${5} "${@:6:10}"
