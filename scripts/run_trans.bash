table_dataset=(
    "synthetic"
    # "synthetic-n15db"
    # "synthetic-n20db"
    # "synthetic-n25db"
    # "synthetic-n30db"
    # "synthetic-n35db"
    # "synthetic-n40db"
    # "measured"
    # "drink-pink"
    # "drink-gold"
    # "drink-pyellow"
    # "drink-blue"
    # "drink-purple"
)


table_model=(
    # "transformer01-wa"
    # "transformer01-wa-nh1"
    # "transformer01-woa"
    # "transformer01-woa-nh1"
    # "transformer02-wa"
    # "transformer02-wa-nh1"
    # "transformer02-woa"
    # "transformer02-woa-nh1"
    # "transformer03-wa"
    # "transformer03-wa-nh1" 
    # "transformer03-woa"
    # "transformer03-woa-nh1"
    # "transformer04-wa"
    # "transformer04-wa-nh1"
    # "transformer04-woa"
    # "transformer04-woa-nh1"
    # "transformer05"
    # "transformer05-nh1"
)

export num_exp=03

export epoch=10
export lr=0.0001
export batch_size=128

for dataset_name in ${table_dataset[@]}; do
    export dataset_name=${dataset_name}
    for model_name in ${table_model[@]}; do
        export model_name=${model_name}
        export run_name="${model_name}-${dataset_name}-${num_exp}"
        bash ~/spec/scripts/template_trans.bash
    done
done

# python plot_results_trans.py