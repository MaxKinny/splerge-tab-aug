dataset=../data/raw/icdar/test/
training_name=${1}

python eval_unofficial.py -i ${dataset}/images -xml ${dataset}/gt -o ${dataset}/ocr -p ./outputs/${training_name}/predicted_xmls -e ./evals/${training_name}
