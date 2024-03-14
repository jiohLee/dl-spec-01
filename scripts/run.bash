table_dataset=(
    "synthetic"
    "synthetic-n15db"
    "synthetic-n20db"
    "synthetic-n25db"
    "synthetic-n30db"
    "synthetic-n35db"
    "synthetic-n40db"
    "measured"
    # "drink-pink"
    # "drink-gold"
    # "drink-pyellow"
    # "drink-blue"
    # "drink-purple"
)


table_model=(
    "cnn-wa"
    "cnn-woa"
    "rescnn-wa"
    "rescnn-woa"
    "deepcubenet1d-wa"
    "deepcubenet1d-woa"
    "resunet-wa"
    "resunet-woa"
    "resunet_cskim-wa"
    "resunet_cskim-woa"
    # "transformer01-wa"
    # "transformer01-woa"
)

export num_exp=01

export epoch=500
export lr=0.0001
export batch_size=1024

for dataset_name in ${table_dataset[@]}; do
    export dataset_name=${dataset_name}
    for model_name in ${table_model[@]}; do
        export model_name=${model_name}
        export run_name="${model_name}-${dataset_name}-${num_exp}"
        bash ~/spec/scripts/template.bash
    done
done

python plot_results.py
