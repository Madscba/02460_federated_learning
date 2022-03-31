#!/bin/bash
## 02460 FL, template
#BSUB -q hpc
#BSUB -J intro_FL_exp
#BSUB -n 20 ##Number of cores
#BSUB -R "rusage[mem=2048MB]"
##BSUB -R "select[model=XeonGold6126]"
#BSUB -R "span[hosts=1]"
#BSUB -M 4GB
#BSUB -W 00:10 ##20 minutes (hh:mm)
###BSUB -B 
#BSUB -N 
#BSUB -o O_fl_%J.out 
#BSUB -e E_fl_%J.err 

#rm -f *.err
#rm -f *.out


filename='/work3/s173934/AdvML/02460_federated_learning/dataset/femnist/data/img_lab_by_user/user_names.txt'
n=1 #spawned_clients
N=20 #amount of clients
n_wait=9
epoch_num=2
exp_id=$(date +"FedAvg_%d%b%T")

echo "starting bash script"

module load python3/3.8.0
source /zhome/87/9/127623/Desktop/env_fl_380/bin/activate


echo "Starting server"
python src/server.py --experiment_id=$exp_id & ##--wandb_user='s173934' wandb_mode="online"
sleep 3  # Sleep for 3s to give the server enough time to start

while read user && (($n<=$N)); do
	echo "Starting client: $n , name: $user"
   	timeout 3m python src/client_main.py --user=${user} --experiment_id=$exp_id --& ##epochs=$epoch_num wandb_mode="online" --wandb_user='s173934'
	if [ $(expr $n % 10) == 0 ] && [ $n > $n_wait ]; then
		echo "sleeping for 90 sec" ##90 sec
		sleep 90
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


