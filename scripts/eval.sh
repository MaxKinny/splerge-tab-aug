dataset=../data/cropped/icdar/test/
training_name=${1}
output_dir=./outputs
eval_dir=./new_evals

python eval.py -i ${dataset}/images -xml ${dataset}/gt -o ${dataset}/ocr -p ${output_dir}/${training_name}/predicted_xmls -e ${eval_dir}/${training_name}

