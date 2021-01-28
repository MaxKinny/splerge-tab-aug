suffix=/icdar
augment=false

root_data_path=../data/cropped
data_path=${root_data_path}/${suffix}/train
val_path=${root_data_path}/${suffix}/val
out_path=$suffix

if [ "$augment" = true ]; then
  out_path=${out_path}-augmented
  args='--augment_tables'
fi

out_path=./trainings/${out_path}_$(date +'%d-%m-%Y')

# rm -r ${out_path}
python train.py --train_dir ${data_path} --val_dir ${val_path} -o ${out_path} -e 250 ${args} --dr 0.8 "${@:1:10}"
