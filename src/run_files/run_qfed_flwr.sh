#!/bin/bash
## 02460 FL, template
#BSUB -q hpc
#BSUB -J intro_FL_exp
#BSUB -n 12 ##Number of cores
#BSUB -R "rusage[mem=2048MB]"
##BSUB -R "select[model=XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -M 4GB
#BSUB -W 03:00 ##20 minutes (hh:mm)
###BSUB -B
#BSUB -N
#BSUB -o o_qfed_flwr.out
#BSUB -e e_qfed_flwr.err


##filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_train.txt'

n=1 #spawned_clients
N=2000 #amount of clients
n_wait=9
##epoch_numbers="1 2 4 8 16 32"
q_param=0.1
##epoch_num=1
rounds=200
wandb_mode="online"
##exp_id1='Qfed_q_param_global'
strategy='Qfed_flwr'
model_name='Qfed_flwr'
epoch_num=8
batch_size=8
num_test_clients=20
one_third_num_test_clients=10 ## you have to do this manually lol
dataset_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'

##exp_id=$(date +"FedAvg_%d%b%T")

echo "starting bash script"

module load python3/3.8.0
source /zhome/fb/d/137704/Desktop/fed_lr/v_env2/bin//activate

echo "Starting server with q param $q_param"
python src/server_main.py \
--wandb_mode=$wandb_mode \
--experiment_id=$strategy$q_param \
--wandb_username='karlulbaek' \
--run_name=$strategy \
--strategy=$strategy \
--q_param=$q_param \
--num_test_clients=$num_test_clients \
--dataset_path=$dataset_path \
--model_name=$model_name \
--config=qfed.yaml \
--entity=karlulbaek \
--api_key=a8ac716e669cdfe0282fc16264fc7533e33e06cf \
--wandb_project=02460_FL \
--rounds=$rounds&pid=$!

sleep 10 # Sleep for 3s to give the server enough time to start

while (($n<=$N)) && ps -p $pid > /dev/null 2>&1; do
  echo "Starting client: $n , name: $n , q param : $q_param"
  python src/client_main.py \
  --seed=$n \
  --qfed=True \
  --config=qfed.yaml\
  --epochs=$epoch_num \
  --batch_size=$batch_size \
  --dataset_path=$dataset_path&

  if [ $(expr $n % 10) == 0 ]; then
    echo "sleeping for" $((30+$one_third_num_test_clients))
    sleep $((30+$one_third_num_test_clients))
  fi
  n=$(($n+1))
done


## This will allow you to use CTRL+C to stop all background processesb
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
## Wait for all background processes to complete
wait


#for i in `seq 0 9`; do
#   echo "Starting client $i"
#   python client.py --user=${i} & >> client.txt
#done

#client_count = 0
#search_dir = /work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/
#for entry in "$search_dir" *.pckl;
#do
#  client_count = client_count+1
#  echo "${entry%.*}"
#done

###ls | sed -n 's/\.pckl$//p' #can extract pckl file names without file extension
###shuf-n 4100 user_names.txt > user_names.txt


