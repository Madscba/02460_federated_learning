import wandb
import sys


parser_default={'max_grad_norm':1.0, 'target_delta':1e-5, 'dp_sgd':False, 'opacus':None, 
            'noise_multiplier':1.0, "sample_rate":0.01,'batch_size':8, 'epochs':1}


def parse_args(parser):
    subparsers = parser.add_subparsers(help='sub-command help')
    parser_a = subparsers.add_parser('wand_config', help='a help')
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
    parser.add_argument('--batch_size', 
        type=int,
        default=8)
    parser.add_argument('--epochs', 
        type=int,
        default=1)
    parser.add_argument("--lib", default=False, help="Set true to use opacus library")
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=0.56,
        metavar="S",
        help="Noise multiplier"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.1
    )
    parser.add_argument(
        "--target_delta",
        type=float,
        default=1e-5
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.01
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
        if key in parser_default.keys():
            if parser_default[key] != val:
                wandb.config.update({key:val}, allow_val_change=True)
                print(f'{key} was overidden with value: {val}', file=sys.stdout)

