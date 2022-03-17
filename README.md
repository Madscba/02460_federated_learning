# 02460_federated_learning

    Project Federated Learning
    
    Description: 
    
    Features: 
    
    How to use: For this project we have used Python 3.7.6. pip install -r requirements.txt to download the libraries used.
    
    Technologies: 

   Collaborators: See commits

   License: MIT License

## Process data

    From inside '/dataset/femnist' run ./preprocess.sh

## Run client

    python src/client_main.py 
    flags 
            --user, default=f0000_14
            --wandb_mode, default='disabled', help=use "online" to log and sync with cloud
            --configs, default=config.yaml help=use fedprox.yaml to run fedprox 
            --experiment_id, default=None type=str  help=parse same experiment id to multiple clients to group into same run
