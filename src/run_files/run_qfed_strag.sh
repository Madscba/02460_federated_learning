#!/bin/bash
## 02460 FL, template
#BSUB -q hpc
#BSUB -J intro_FL_exp
#BSUB -n 24 ##Number of cores
#BSUB -R "rusage[mem=2048MB]"
##BSUB -R "select[model=XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -M 4GB
#BSUB -W 05:00 ##20 minutes (hh:mm)
###BSUB -B
#BSUB -N
#BSUB -o mlrtest.out
#BSUB -e mlrtest.err


##filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_train.txt'

n=1 #spawned_clients
N=100000 #amount of clients
n_wait=9
##epoch_numbers="1 2 4 8 16 32"
q_param=0.1
##epoch_num=1
rounds=1
wandb_mode="disabled"
##exp_id1='Qfed_q_param_global'
strategy='Qfed_manual'
model_name='Qfed_mlr_test'
epoch_num=1
batch_size=32
model='mlr'
num_classes=10
lr=0.05
num_test_clients=20
one_third_num_test_clients=0 ## you have to do this manually lol
dataset_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'

##exp_id=$(date +"FedAvg_%d%b%T")

echo "starting bash script"

module load python3/3.8.0
source /zhome/fb/d/137704/Desktop/fed_lr/v_env2/bin//activate

echo "Starting server with q param $q_param"
python src/server_main.py \
--wandb_mode=$wandb_mode \
--model=$model \
--experiment_id=$model_name$q_param \
--wandb_username='karlulbaek' \
--run_name=$strategy \
--num_test_clients=$num_test_clients \
--strategy=$strategy \
--q_param=$q_param \
--model_name=$model_name \
--dataset_path=$dataset_path \
--config=qfed.yaml \
--entity=karlulbaek \
--api_key=a8ac716e669cdfe0282fc16264fc7533e33e06cf \
--wandb_project=02460_FL \
--rounds=$rounds&pid=$!

sleep 10 # Sleep for 3s to give the server enough time to start

while read user && (($n<=$N)); do

	if [ $(expr $n % 10) == 0 ] && [ $n<=10 ]
		echo "Starting client: $(n) , name: $user (straggler)"
  	echo "Starting client: $n , name: $n , q param : $q_param"
      timeout 2m python src/client_main.py \
    --seed=$n \
    --qfed=True \
    --model=$model \
    --config=qfed.yaml \
    --num_classes=$num_classes \
    --epochs=1 \
    --batch_size=$batch_size \
    --lr=$lr \
    --dataset_path=$dataset_path&
  else
		echo "Starting client: $n , name: $user"
   	echo "Starting client: $n , name: $n , q param : $q_param"
      timeout 2m python src/client_main.py \
    --seed=$n \
    --qfed=True \
    --model=$model \
    --config=qfed.yaml \
    --num_classes=$num_classes \
    --epochs=$epoch_num \
    --batch_size=$batch_size \
    --lr=$lr \
    --dataset_path=$dataset_path&
  fi
	if [ $(expr $n % 10) == 0 ] && [ $n>$n_wait ]; then
		echo "sleeping for 15 sec" ##120 sec
		sleep 15
	fi
	n=$((n+1))
done < $filename



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


