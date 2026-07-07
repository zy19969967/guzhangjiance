import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='From https://github.com/Feaxure-fresh/TL-Fault-Diagnosis-Library')
 
    # basic parameters
    parser.add_argument('--model_name', type=str, default='CNN',
                        help='Name of the model (in ./models directory)')
    parser.add_argument('--source', type=str, default='CWRU_0',
                        help='Source data, separated by "," (select specific conditions of the dataset with name_number, such as CWRU_0)')
    parser.add_argument('--target', type=str, default='CWRU_1',
                        help='Target data (select specific conditions of the dataset with name_number, such as CWRU_0)')
    parser.add_argument('--data_dir', type=str, default="./datasets",
                        help='Directory of the datasets')
    parser.add_argument('--train_mode', type=str, default='single_source',
                        choices=['single_source', 'source_combine', 'multi_source'],
                        help='Training mode (select correctly before training)')
    parser.add_argument('--cuda_device', type=str, default='0',
                        help='Allocate the device to use only one GPU ('' means using cpu)')
    parser.add_argument('--max_epoch', type=int, default=30,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--signal_size', type=int, default=1024,
                        help='Signal length split by sliding window')
    parser.add_argument('--stride', type=int, default=0,
                        help='Sliding-window stride. 0 means non-overlap and equals signal_size')
    parser.add_argument('--in_channel', type=int, default=1,
                        help='Number of input signal channels')
    parser.add_argument('--random_state', type=int, default=10,
                        help='Random state for the entire training')

    # optimizer parameters          
    parser.add_argument('--opt', type=str, choices=['sgd', 'adam'], default='sgd', help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for sgd')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Betas for adam')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for both sgd and adam')
   
    # learning rate parameters
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate')
    parser.add_argument('--lr_scheduler', type=str, choices=['step', 'exp', 'stepLR', 'fix'], default='stepLR',
                        help='Type of learning rate schedule')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Parameter for the learning rate scheduler (except "fix")')
    parser.add_argument('--steps', type=str, default='10',
                        help='Step of learning rate decay for "step" and "stepLR"')
    
    # optimization parameters
    parser.add_argument('--backbone', type=str, default='CNN', choices=['CNN', 'ResNet'],
                        help='The backbone used to construct the training model (defined in ./modules)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for dataloader')
    parser.add_argument('--normlize_type', type=str, choices=['0-1', '-1-1', 'mean-std'], default='-1-1',
                        help='Data normalization methods')
    parser.add_argument('--target_eval_mode', type=str, choices=['full', 'split'], default='full',
                        help='Use full target domain for validation, or split target into train/val')
    parser.add_argument('--tradeoff', type=list, default=['exp', 'exp', 'exp'],
                        help='Trade-off coefficients for the sum of loss terms. Use integer or "exp" ("exp" represents an increase from 0 to 1 during training)')
    parser.add_argument('--zeta', type=float, default=10.0,
                        help='Parameter to control the increasing rate of "exp" tradeoff')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout layer coefficient')

    # MSSA_PLUS parameters
    parser.add_argument('--source_ema_momentum', type=float, default=0.95,
                        help='EMA momentum for MSSA_PLUS source reliability weighting')
    parser.add_argument('--pseudo_temperature', type=float, default=0.5,
                        help='Sharpening temperature for MSSA_PLUS ensemble pseudo labels')
    parser.add_argument('--source_weight_tau', type=float, default=2.0,
                        help='Softmax temperature for MSSA_PLUS source reliability weights')
    parser.add_argument('--consistency_weight', type=float, default=0.2,
                        help='Target classifier-consistency weight for MSSA_PLUS')
    parser.add_argument('--mcc_weight', type=float, default=0.1,
                        help='Target minimum class confusion weight for MSSA_PLUS')
    parser.add_argument('--pseudo_threshold', type=float, default=0.0,
                        help='Confidence threshold for MSSA_PLUS pseudo-label weighting')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Source classification label smoothing for MSSA_PLUS')
    parser.add_argument('--axis_blend', type=float, default=0.35,
                        help='Blend weight for MSSA_PLUS_HYBRID axis-derived class probabilities')
    parser.add_argument('--axis_loss_weight', type=float, default=0.3,
                        help='Source axis BCE loss weight for MSSA_PLUS_HYBRID')
    parser.add_argument('--axis_nll_weight', type=float, default=0.2,
                        help='Source axis-combination NLL loss weight for MSSA_PLUS_HYBRID')
    parser.add_argument('--head_cons_weight', type=float, default=0.1,
                        help='Source class-head/axis-head consistency weight for MSSA_PLUS_HYBRID')
    parser.add_argument('--target_head_cons_weight', type=float, default=0.05,
                        help='Target class-head/axis-head consistency weight for MSSA_PLUS_HYBRID')
    
    # save and load
    parser.add_argument('--save', type=bool, default=True, help='Save logs and trained model checkpoints')
    parser.add_argument('--save_dir', type=str, default='./ckpt',
                        help='Directory to save logs and model checkpoints')
    parser.add_argument('--load_path', type=str, default='',
                        help='Load trained model checkpoints from this path (for testing, not for resuming training)')
    args = parser.parse_args()
    return args
    

