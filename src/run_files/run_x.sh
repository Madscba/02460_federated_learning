#!/bin/bash
## 02460 FL, template
#BSUB -q hpc
#BSUB -J fedx
#BSUB -n 12 ##Number of cores
#BSUB -R "rusage[mem=2048MB]"
##BSUB -R "select[model=XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -M 4GB
#BSUB -W 06:00 ##20 minutes (hh:mm)
###BSUB -B 
#BSUB -N 
#BSUB -o O_fl_mlr_batch_%J.out 
#BSUB -e E_fl_mlr_batch_%J.err 


##filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_train.txt'
dataset_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'
n=1 #spawned_clients
s=1
n_stragglers=5
N=200000 #amount of clients
n_wait=9
batch_size=8
epoch_num=16
num_classes=10
lr=0.001
rounds=300
wandb_mode="online"
exp_id='FedX_hyper_high'
strategy='FedX'
model='mlr'
config=fedx.yaml
##exp_id=$(date +"FedAvg_%d%b%T")

echo "starting bash script"

module load python3/3.8.0
source /zhome/dd/4/128822/fl_380/bin/activate

##sigmas="0.0001"
##norms="20.0"
norms="10.0 1.0"
##mus="1.0"
qs="0.01"
##q_val=0.01
sigma=0.0001
##max_grad=0.1
mu=1


for q_val in $qs
do
	for max_grad in $norms
	do
		run_name="X_(sigma=$sigma)_(S2=$max_grad)_(mu=$mu)_(q=$q_val)"
		echo "Starting server $run_name"
		python src/server_main.py \
		--strategy=$strategy \
		--wandb_mode=$wandb_mode \
		--model=$model \
		--model_name=$run_name \
		--experiment_id=$exp_id \
		--run_name=$run_name \
		--wandb_username='johannes_boe' \
		--entity johannes_boe \
		--api_key d9a0e4bbe478bc7e59b80f931b0281cb3501e8dd \
		--wandb_project 02460_FL \
		--configs=$config \
		--q_param=$q_val \
		--noise_multiplier=$sigma \
		--max_grad_norm=$max_grad \
		--total_num_clients=3000 \
		--rounds=$rounds&pid=$!

		sleep 5 # Sleep for 3s to give the server enough time to start

		while (($n<=$N)) && ps -p $pid > /dev/null 2>&1; do
			if [ "$s" -le "$n_stragglers" ]
			then 
				if [ $drop_stragglers == "true" ]; then :;
				else
					echo "Starting client: $n,  name: $user (straggler)"
					python src/client_main.py \
					--seed=$n \
					--model=$model \
					--qfed=True \
					--experiment_id=$exp_id \
					--epochs=1 \
					--mu $mu \
					--num_classes $num_classes \
					--wandb_mode="disabled" \
					--wandb_username='johannes_boe' \
					--entity johannes_boe \
					--api_key d9a0e4bbe478bc7e59b80f931b0281cb3501e8dd \
					--wandb_project 02460_FL \
					--job_type="client_$strategy" \
					--configs=$config \
					--batch_size=$batch_size \
					--lr=$lr\
					--max_grad_norm=$max_grad \
					--dataset_path=$dataset_path& 
				fi
			else
				echo "Starting client: $n,  name: $user"
				python src/client_main.py \
				--seed=$n \
				--model=$model \
				--qfed=True \
				--experiment_id=$exp_id \
				--epochs=$epoch_num \
				--mu $mu \
				--num_classes $num_classes \
				--wandb_mode="disabled" \
				--wandb_username='johannes_boe' \
				--entity johannes_boe \
				--api_key d9a0e4bbe478bc7e59b80f931b0281cb3501e8dd \
				--wandb_project 02460_FL \
				--job_type="client_$strategy" \
				--configs=$config \
				--batch_size=$batch_size \
				--lr=$lr\
				--max_grad_norm=$max_grad \
				--dataset_path=$dataset_path& 
			fi
			if [ $(expr $n % 10) == 0 ]; then
				echo "sleeping for" $((5))
				sleep $((5)) ##$((30+5*$epoch_num))
				s=0
			fi
			s=$(($s+1))
			n=$(($n+1))
		done
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


