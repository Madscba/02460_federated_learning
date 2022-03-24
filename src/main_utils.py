import wandb


parser_default={'max_grad_norm':1.0, 'target_delta':1e-5, 'dp_sgd':False, 'opacus':False, 
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
        default=8)
    parser.add_argument('--epochs', 
        default=1)

    parser.add_argument("--dp_sgd",
        default=False)
    parser.add_argument("--opacus", 
        default=None, 
        help="Set true to use opacus library")
    parser.add_argument(
        "--noise_multiplier",
        type=float,
        default=1.0,
        metavar="S",
        help="Noise multiplier",
    )
    parser.add_argument(
        "-c",
        "--max_grad_norm",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--target_delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.01,
        metavar="Q",
        help="Sample rate (determined on server level)",
    )

    args = parser.parse_args()
    return args

def update_config(args):
    for key, val in args._get_kwargs():
        if key in parser_default.keys():
            if parser_default[key] != val:
                wandb.config[key]=val

