#!/bin/bash
## 02460 FL, template
#BSUB -q hpc
#BSUB -J intro_FL_exp
#BSUB -n 20 ##Number of cores
#BSUB -R "rusage[mem=2048MB]"
##BSUB -R "select[model=XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -M 4GB
#BSUB -W 10:00 ##20 minutes (hh:mm)
###BSUB -B 
#BSUB -N 
#BSUB -o O_fl_%J.out 
#BSUB -e E_fl_%J.err 


filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_train.txt'
n=1 #spawned_clients
s=1
N=10 #amount of clients
epoch_num=20
rounds=200
strategy='FedAvg'
wandb_mode='online'
n_stragglers=5
exp_id='FedProx_vs_FedAvg'
config=config.yaml
num_classes=10
model='Net'
drop_stragglers="true"
#job_type="server_straggler_($n_stragglers)/10_E($epoch_num)" 
job_type="server"

echo "starting bash script"

module load python3/3.8.0
source /zhome/db/f/128823/Desktop/fl_362/bin/activate

echo "Starting server"
python src/server_main.py --wandb_mode=$wandb_mode \
--experiment_id=$exp_id \
--wandb_username='s175548' \
--run_name="($strategy)_($model)_($num_classes)c_drop_$drop_stragglers" \
--model $model \
--job_type $job_type \
--entity s175548 \
--api_key 47304b319fc295d13e84bba0d4d020fc41bd0629 \
--wandb_project 02460_federated_learning \
--rounds $rounds&pid=$!
sleep 15  # Sleep for 3s to give the server enough time to start
while read user && (($n<=$N)) && ps -p $pid > /dev/null 2>&1; do
	if [ "$s" -le "$n_stragglers" ]
	then	
		if [ $drop_stragglers == "true" ]; then :;
		else
			echo "Starting client: $n , name: $user (straggler)"
			python src/client_main.py --seed=$n --wandb_username='s175548' \
			--job_type="client_$strategy 2" \
			--wandb_mode='disabled' \
			--experiment_id=$exp_id \
			--configs=$config \
			--epochs=1 \
			--num_classes $num_classes \
			--entity s175548 \
			--api_key 47304b319fc295d13e84bba0d4d020fc41bd0629 \
			--model $model \
			--dataset_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'&
		fi  
	else
		echo "Starting client: $n , name: $user"
   		python src/client_main.py --seed=$n --wandb_username='s175548' \
		--job_type="client_$strategy 2" \
		--wandb_mode='disabled' \
		--experiment_id=$exp_id \
		--configs=$config \
		--epochs=$epoch_num \
		--num_classes $num_classes \
		--entity s175548 \
		--api_key 47304b319fc295d13e84bba0d4d020fc41bd0629 \
		--wandb_project 02460_federated_learning \
		--model $model \
		--dataset_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'& 
	fi
	if [ $(expr $n % 10) == 0 ]; then
		echo "sleeping for 100  sec" ## sec
		sleep 100
		s=0
	fi
	s=$(($s+1))
	n=$(($n+1))
done < $filename


## This will allow you to use CTRL+C to stop all background processesb
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
## Wait for all background processes to complete
wait
