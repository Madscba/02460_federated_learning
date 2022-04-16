#!/bin/bash
## 02460 FL, template
#BSUB -q hpc
#BSUB -J intro_FL_exp
#BSUB -n 12 ##Number of cores
#BSUB -R "rusage[mem=2048MB]"
##BSUB -R "select[model=XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -M 4GB
#BSUB -W 24:00 ##20 minutes (hh:mm)
###BSUB -B 
#BSUB -N 
#BSUB -o O_fl_lr_%J.out 
#BSUB -e E_fl_lr_%J.err 


##filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_train.txt'

n=1 #spawned_clients
N=30000 #amount of clients
n_wait=9
lr_rates="0.0001 0.001 0.01 0.1"
epoch_num=1
rounds=200
wandb_mode="online"
exp_id='FedAvg_lr'
strategy='FedAvg'
##exp_id=$(date +"FedAvg_%d%b%T")

echo "starting bash script"

module load python3/3.8.0
source /zhome/87/9/127623/Desktop/env_fl_380/bin/activate

for lr_rate in $lr_rates
do
	echo "Starting server $epoch_num"
	python src/server_main.py --wandb_mode=$wandb_mode \
	--experiment_id=$exp_id$lr_rate \
	--wandb_username='s173934' \
	--run_name=$strategy \
	--entity madscba \
	--api_key a49a6933370e2c529423c7f224c5e773600b033b \
	--wandb_project 02460_FL \
	--rounds=$rounds&pid=$!

	sleep 3 # Sleep for 3s to give the server enough time to start

	while (($n<=$N)) && ps -p $pid > /dev/null 2>&1; do
		echo "Starting client: $n, lr: $lr_rate"
	   	timeout 2m python src/client_main.py \
		--seed=$n \
		--experiment_id=$exp_id$lr_rate \
		--epochs=$epoch_num \
		--wandb_mode=$wandb_mode \
		--wandb_username='s173934' \
		--entity madscba \
		--api_key a49a6933370e2c529423c7f224c5e773600b033b \
		--wandb_project 02460_FL \
		--job_type="client_$strategy" \
		--config=config.yaml\
		--lr=$lr_rate\
		 --dataset_path=$datapath& 

		if [ $(expr $n % 10) == 0 ]; then
			echo "sleeping for 30s" 
			sleep 30 ##$((30+5*$epoch_num))
		fi
		n=$(($n+1))
	done
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


