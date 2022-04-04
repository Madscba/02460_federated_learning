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


filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/user_names_train.txt'
n=1 #spawned_clients
N=1000 #amount of clients
epoch_num=20
n_wait=10
straggler_pct=0.5
exp_id='FedProx_vs_FedAvg'

echo "starting bash script"

module load python3/3.8.0
source /zhome/db/f/128823/Desktop/fl_362/bin/activate

echo "Starting server"
python src/server.py --wandb_mode=online --experiment_id=$exp_id --wandb_username='s175548' --run_name='FedProx' --rounds 200 &
sleep 3  # Sleep for 3s to give the server enough time to start
while read user && (($n<=$N)); do
	if [ $(expr $n % 2) == 0 ]
	then	
		echo "Starting client: $n , name: $user (straggler)"
		timeout 6m python src/client_main.py --user=${user} --wandb_username='s175548' --job_type='client_prox' --wandb_mode="disabled" --experiment_id=$exp_id --configs=fedprox.yaml --epochs=1 --dataset_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'& 
	else
		echo "Starting client: $n , name: $user"
   		timeout 6m python src/client_main.py --user=${user} --wandb_username='s175548' --job_type='client_prox' --wandb_mode="disabled" --experiment_id=$exp_id --configs=fedprox.yaml --epochs=$epoch_num --dataset_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'& 
	fi
	if [ $(expr $n % 10) == 0 ]; then
		echo "sleeping for 180 sec" ##90 sec
		sleep 180
	fi
	n=$(($n+1))
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

