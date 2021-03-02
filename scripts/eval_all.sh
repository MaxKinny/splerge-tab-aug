dataset=../data/cropped/icdar/test/

# folder="1"
for folder in $(ls trainings/); do
	for training_name in $(ls trainings/${folder}/); do
		train=./trainings/${folder}/${training_name}
		output=./outputs/${folder}/${training_name}
		eval=./new_evals/${folder}/${training_name}
		python infer.py -img ${dataset}/images -m ${train}/last_model.pth -o ${output}/
		python eval.py -i ${dataset}/images -xml ${dataset}/gt -o ${dataset}/ocr -p ${output}/predicted_xmls -e ${eval}/
	done
done
