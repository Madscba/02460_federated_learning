#!/bin/bash
## 02460 FL, template
#BSUB -q hpc
#BSUB -J fedx_ex5
#BSUB -n 12 ##Number of cores
#BSUB -R "rusage[mem=2048MB]"
##BSUB -R "select[model=XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -M 4GB
#BSUB -W 01:30 ##120 minutes (hh:mm)
###BSUB -B 
#BSUB -N 
#BSUB -o O_fl_%J.out 
#BSUB -e E_fl_%J.err 

#rm -f *.err
#rm -f *.out


filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_train.txt'
n=1 #spawned_clients
N=2950 #amount of clients
n_wait=9
epoch_num=8
rounds=30
#num_classes=20
wandb_mode="online"
exp_id='FedX'
strategy='FedX'
batch_size=8
straggler_pct=0.5

echo "starting bash script"

module load python3/3.8.0
source /zhome/dd/4/128822/fl_380/bin/activate

echo "Starting server"
qs="0.0001 0.001 0.002"
sigma=0.005
norms="0.1 1.0 10.0"

python src/server_main.py --wandb_mode=$wandb_mode \
--strategy=$strategy \
--experiment_id=$exp_id \
--wandb_username='johannes_boe' \
--run_name=$strategy \
--entity johannes_boe \
--api_key d9a0e4bbe478bc7e59b80f931b0281cb3501e8dd \
--wandb_project 02460_FL \
--configs=fedx.yaml \
--noise_multiplier=0.001 \
--max_grad_norm=10.0 \
--q_param=0.001 \
--batch_size=$batch_size \
--total_num_clients=$N \
--rounds=$rounds&pid=$!

sleep 3  # Sleep for 3s to give the server enough time to start


while read user && (($n<=$N)) && ps -p $pid > /dev/null 2>&1; do
	echo "Starting client: $n , name: $user (straggler)"
	python src/client_main.py \
	--seed=$n \
	--experiment_id=$exp_id \
	--epochs=$epoch_num \
	--wandb_mode="disabled" \
	--wandb_username='johannes_boe' \
	--job_type="client_$strategy" \
	--config=fedx.yaml \
	--epochs=$epoch_num \
	--entity johannes_boe \
	--api_key d9a0e4bbe478bc7e59b80f931b0281cb3501e8dd \
	--wandb_project 02460_FL \
	--batch_size=$batch_size \
	--qfed=True \
	--dataset_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'&
	if [ $(expr $n % 10) == 0 ]; then
		echo "sleeping for" $((30+5*$epoch_num)) 
		sleep $((30+5*$epoch_num))
	fi
	n=$(($n+1))
done < $filename


## This will allow you to use CTRL+C to stop all background processesb
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
## Wait for all background processes to complete
wait


