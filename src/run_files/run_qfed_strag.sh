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
#BSUB -o strag4.out
#BSUB -e strag4.err


##filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_train.txt'

n=1 #spawned_clients
N=100000 #amount of clients
n_wait=9
##epoch_numbers="1 2 4 8 16 32"
q_param=0.01
drop_stragglers="false"
n_stragglers=5
##epoch_num=1
rounds=1000
wandb_mode="online"
##exp_id1='Qfed_q_param_global'
strategy='Qfed_manual'
model_name='qfed_strag_1000_rounds'
epoch_num=16
batch_size=16
model='mlr'
num_classes=10
lr=0.0001
num_test_clients=20
one_third_num_test_clients=0 ## you have to do this manually lol
dataset_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'
filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_train.txt'

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
--entity=02460-fl \
--api_key=a8ac716e669cdfe0282fc16264fc7533e33e06cf \
--wandb_project=final_experiments \
--rounds=$rounds&pid=$!
sleep 15 # Sleep for 3s to give the server enough time to start
while (($n<=$N)) && ps -p $pid > /dev/null 2>&1; do
	if [ "$s" -le "$n_stragglers" ]
	then
		if [ $drop_stragglers == "true" ]; then :;
		else
			echo "Starting client: $n , name: $user (straggler)"
      python src/client_main.py \
      --seed=$n \
      --qfed=True \
      --model=$model \
      --config=qfed.yaml \
      --num_classes=$num_classes \
      --epochs=1 \
      --batch_size=$batch_size \
      --lr=$lr \
      --dataset_path=$dataset_path&
		fi
	else
		echo "Starting client: $n , name: $user"
    python src/client_main.py \
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
	if [ $(expr $n % 10) == 0 ]; then
		echo "sleeping for 3  sec" ## sec
		sleep 3
		s=0
	fi
	s=$(($s+1))
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


