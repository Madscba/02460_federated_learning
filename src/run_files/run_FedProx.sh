#!/bin/bash
## 02460 FL, template
#BSUB -q hpc
#BSUB -J intro_FL_exp
#BSUB -n 11 ##Number of cores
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


filename='/zhome/87/9/127623/Desktop/02460_federated_learning/dataset/femnist/data/img_lab_by_user/user_names.txt'
n=1 #spawned_clients
N=500 #amount of clients
epoch_num=2
n_wait=10
straggler_pct=0.5
exp_id=$(date +"FedProx_%d%b%T")

echo "starting bash script"

module load python3/3.6.2
source /zhome/87/9/127623/Desktop/env_fl/bin/activate


echo "Starting server"
python src/server_main.py &
sleep 3  # Sleep for 3s to give the server enough time to start


while read user && (($n<=$N)); do
	
	if [ $(expr $n % 10) == 0 ] && [ $n<=10 ]	
		echo "Starting client: $(n) , name: $user (straggler)"
  		timeout 4m python src/client_main.py --user=${user} --wandb_mode="online" --experiment_id=$exp_id --configs=fedprox.yaml --epochs=$($straggler_pct*$epoch_num)&
	else
		echo "Starting client: $n , name: $user"
   		timeout 4m python src/client_main.py --user=${user} --wandb_mode="online" --experiment_id=$exp_id --configs=fedprox.yaml --epochs=$epoch_num&
		
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



