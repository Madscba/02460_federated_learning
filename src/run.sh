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


##Taken from https://stackoverflow.com/questions/192249/how-do-i-parse-command-line-arguments-in-bash
echo start

rm -f *.err
rm -f *.out
echo "starting"

module load python3/3.6.2
source /zhome/87/9/127623/Desktop/env_fl/bin/activate

echo "Starting server"
python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 9`; do
   echo "Starting client $i"
   python client.py --partition=${i} & | grep client.txt
done

## This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
## Wait for all background processes to complete
wait
