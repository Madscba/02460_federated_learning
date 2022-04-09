#!/bin/bash
## 02460 FL, template
#BSUB -q hpc
#BSUB -J intro_FL_exp
#BSUB -n 20 ##Number of cores
#BSUB -R "rusage[mem=2048MB]"
##BSUB -R "select[model=XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -M 4GB
#BSUB -W 02:30 ##20 minutes (hh:mm)
###BSUB -B 
#BSUB -N 
#BSUB -o O_fl_%J.out 
#BSUB -e E_fl_%J.err 


filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_train.txt'
datapath='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'
n=1 #spawned_clients
N=2950 #amount of clients
n_wait=9
epoch_num=2
rounds=200
wandb_mode="online"
exp_id='Qfed_manual'
strategy='Qfed_manual'
##exp_id=$(date +"FedAvg_%d%b%T")

echo "starting bash script"

module load python3/3.8.0
source /zhome/87/9/127623/Desktop/env_fl_380/bin/activate


echo "Starting server"
python src/server.py --wandb_mode=$wandb_mode --experiment_id=$exp_id --wandb_username='s173934' --run_name=$strategy --rounds=$rounds&pid=$!
sleep 3 # Sleep for 3s to give the server enough time to start

while read user && (($n<=$N)) && ps -p $pid > /dev/null 2>&1; do
	echo "Starting client: $n , name: $user"
   	timeout 2m python src/client_main.py --user=${user} --experiment_id=$exp_id --epochs=$epoch_num --wandb_mode=$wandb_mode --wandb_username='s173934' --job_type="client_$strategy" --config=config.yaml --dataset_path=$datapath --qfed=True& 
	if [ $(expr $n % 10) == 0 ]; then
		echo "sleeping for 60 sec" ##90 sec
		sleep 60
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


