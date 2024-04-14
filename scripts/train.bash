table_dataset=(
    # "synthetic"
    # "synthetic-n40db"
    # "synthetic-n30db"
    # "synthetic-n20db"
    # "synthetic-n35db"
    # "synthetic-n25db"
    # "synthetic-n15db"
    "measured"
)


table_model=(
    "cnn-wa"
    "cnn-wagu"
    "cnn-woa"
    # "rescnn-wa"
    # "rescnn-wagu"
    # "rescnn-woa"
    # "resunet-wa"
    # "resunet-wagu"
    # "resunet-woa"
)

export num_exp=01

export epoch=10
export lr=0.0001
export batch_size=1024

for dataset_name in ${table_dataset[@]}; do
    export dataset_name=${dataset_name}
    for model_name in ${table_model[@]}; do
        export model_name=${model_name}
        export run_name="${model_name}-${dataset_name}-${num_exp}"
        bash ~/spec/scripts/train_template.bash
    done
done

