gpus=$1    # 0,1
maps=$2    # MMM2, 3s5z
alg=$3     # vdn
args=$4    # rnd_belta=0.1

args=(${args//,/ })

export CUDA_VISIBLE_DEVICES="$gpus"
python src/main.py --config="$alg" --env-config=sc2 with env_args.map_name="$maps" "${args[@]}"