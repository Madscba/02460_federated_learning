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
n=1
exp_id=date

echo "starting bash script"

module load python3/3.6.2
source /zhome/87/9/127623/Desktop/env_fl/bin/activate


echo "Starting server"
python src/server.py &
sleep 3  # Sleep for 3s to give the server enough time to start


while read user && (($n<=20)); do
	echo "Starting client: $n , name: $user"
   	python src/client_main.py --user=${user} --wandb_mode="online" --experiment_id=$exp_id&
	n=$((n+1))
	##if [ $(expr $n % 20) == 0 ]
	##then
##		echo "sleeping for 5 sec"
##		sleep 5
##	fi
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


