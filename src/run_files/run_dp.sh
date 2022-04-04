#!/bin/bash
## 02460 FL, template
#BSUB -q hpc
#BSUB -J dp_exp
#BSUB -n 2 ##Number of cores
#BSUB -R "rusage[mem=2048MB]"
##BSUB -R "select[model=XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -M 4GB
#BSUB -W 00:25 ##20 minutes (hh:mm)
###BSUB -B 
#BSUB -N 
#BSUB -o O_fl_%J.out 
#BSUB -e E_fl_%J.err 

#rm -f *.err
#rm -f *.out


filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/usernames_train.txt'
n=1 #spawned_clients
N=500 #amount of clients
n_wait=9
epoch_num=2
exp_id=$(date +"DP_%d%b%T")

echo "starting bash script"

module load python3/3.8.0
source /zhome/dd/4/128822/fl_362/bin/activate

echo "Starting server"
python src/server.py --strategy="DP_Fed" --experiment_id=$exp_id --wandb_username='johannes_boe' --configs=dp_sgd.yaml &
sleep 3  # Sleep for 3s to give the server enough time to start


while read user && (($n<=$N)); do
	echo "Starting client: $n , name: $user"
   	timeout 4m python src/client_main.py --user=${user} --experiment_id=$exp_id --wandb_username='johannes_boe' --configs=dp_sgd.yaml --epochs=$epoch_num --dataset_path='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist'& 
	if [ $(expr $n % 10) == 0 ] && [ $n>$n_wait ]; then
		echo "sleeping for 120 sec" ##120 sec
		sleep 120
	fi
	n=$((n+1))
done < $filename


## This will allow you to use CTRL+C to stop all background processesb
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
## Wait for all background processes to complete
wait


