import wandb
import sys


def parse_args(parser):
    parser.add_argument('--user', 
                help='user ex f0000_14', default='f0000_14')
    parser.add_argument('--wandb_mode', 
                help='use "online" to log and sync with cloud', default='disabled')
    parser.add_argument('--configs', 
        default='config.yaml')
    
    parser.add_argument('--experiment_id', 
        default=None)
    parser.add_argument('--wandb_username', 
        default=None)
    parser.add_argument('--job_type', 
        default='client')
    parser.add_argument('--dataset_path',
        default=None)

    #arguments used to overwrite config files

    parser.add_argument('--batch_size', 
        type=int,
        default=8)
    parser.add_argument('--epochs', 
        type=int,
        default=1)
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.1,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.90
    )
    parser.add_argument(
        "--qfed",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--straggler",
        type=bool,
        default=False
    )
    args = parser.parse_args()
    return args

def update_config(args):
    for key, val in args._get_kwargs():
        if key in wandb.config._items.keys():
            if wandb.config[key] != val:
                wandb.config.update({key:val}, allow_val_change=True)
                print(f'{key} was overidden with value: {val}', file=sys.stdout)

