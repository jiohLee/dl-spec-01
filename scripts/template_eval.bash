time python eval.py \
--model_name ${model_name} \
--dataset_name drink-pink \
--run_name ${run_name}-drink-pink \
--weights /root/dl-spec-01/results/${model_name}-measured-${num_exp}/${model_name}-measured-${num_exp}.pt \

time python eval.py \
--model_name ${model_name} \
--dataset_name drink-gold \
--run_name ${run_name}-drink-gold \
--weights /root/dl-spec-01/results/${model_name}-measured-${num_exp}/${model_name}-measured-${num_exp}.pt \

time python eval.py \
--model_name ${model_name} \
--dataset_name drink-pyellow \
--run_name ${run_name}-drink-pyellow \
--weights /root/dl-spec-01/results/${model_name}-measured-${num_exp}/${model_name}-measured-${num_exp}.pt \

time python eval.py \
--model_name ${model_name} \
--dataset_name drink-blue \
--run_name ${run_name}-drink-blue \
--weights /root/dl-spec-01/results/${model_name}-measured-${num_exp}/${model_name}-measured-${num_exp}.pt \

time python eval.py \
--model_name ${model_name} \
--dataset_name drink-purple \
--run_name ${run_name}-drink-purple \
--weights /root/dl-spec-01/results/${model_name}-measured-${num_exp}/${model_name}-measured-${num_exp}.pt \