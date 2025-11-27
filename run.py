import argparse
import os
from utils.print_args import print_args
from utils.str2bool import str2bool
import random
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='TimesNet')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')

    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--conv_kernel', nargs='+', type=int, default=[24], help='conv kernel size for MICN')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    ### Add some args for each model
    # Mamba
    parser.add_argument('--save_log', type=str2bool, default=False, help='whether to save log for MambaSingleLayer')
    parser.add_argument('--save_csv', type=str2bool, default=False, help='whether to save csv result for MambaSingleLayer')
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--tv_dt', type=int, default=0, help='whether to use time variant dt for Mamba')
    parser.add_argument('--tv_B', type=int, default=0, help='whether to use time variant B for Mamba')
    parser.add_argument('--tv_C', type=int, default=0, help='whether to use time variant C for Mamba')
    parser.add_argument('--use_D', type=int, default=0, help='whether to use D for Mamba')
    parser.add_argument('--mamba_projection_type', type=str, default='average', 
                        help='select projection/aggregation method for Mamba ablation study, options: [last, average, gating, flatten]')
    
    # DLinear
    parser.add_argument('--individual', type=str2bool, default=False, help='DLinear: a linear layer for each variate(channel) individually')
    # +) moving_avg

    # LightTS
    parser.add_argument('--chunk_size', type=int, default=24, help='subsequence size for LightTS')
    # +) d_model

    # MTSMixer
    parser.add_argument('--fac_T', type=str2bool, default=False, help='whether to apply factorized temporal interaction')
    parser.add_argument('--fac_C', type=str2bool, default=False, help='whether to apply factorized channel interaction')
    parser.add_argument('--use_revin', type=str2bool, default=False, help='whether to apply RevIN')
    # +) e_layers, d_model, d_ff, use_norm, down_sampling_window, individual

    # TimesNet
    # +) e_layers, d_model, d_ff, num_kernels, top_k

    # ModernTCN
    parser.add_argument('--stem_ratio', type=int, default=6, help='stem ratio')
    parser.add_argument('--downsample_ratio', type=int, default=2, help='downsample_ratio')
    parser.add_argument('--ffn_ratio', type=int, default=2, help='ffn_ratio')
    parser.add_argument('--num_blocks', nargs='+',type=int, default=[1,1,1,1], help='num_blocks in each stage')
    parser.add_argument('--large_size', nargs='+',type=int, default=[31,29,27,13], help='big kernel size')
    parser.add_argument('--small_size', nargs='+',type=int, default=[5,5,5,5], help='small kernel size for structral reparam')
    parser.add_argument('--dims', nargs='+',type=int, default=[256,256,256,256], help='dmodels in each stage')
    parser.add_argument('--dw_dims', nargs='+',type=int, default=[256,256,256,256], help='dw dims in dw conv in each stage')
    parser.add_argument('--small_kernel_merged', type=str2bool, default=False, help='small_kernel has already merged or not')
    parser.add_argument('--call_structural_reparam', type=bool, default=False, help='structural_reparam after training')
    parser.add_argument('--use_multi_scale', type=str2bool, default=False, help='use_multi_scale fusion')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--decomp_moderntcn', type=int, default=0, help='decomposition; True 1 False 0')
    parser.add_argument('--decomp_kernel_size', type=int, default=25, help='decomposition-kernel')
    # +) patch_size, patch_stride, use_revin

    # TimeMixerPP
    parser.add_argument('--channel_mixing', type=int, default=1,
                        help='0: channel mixing 1: whether to use channel_mixing')
    parser.add_argument('--output_attention', type=str2bool, default=False, help='whether to output attention in ecoder')
    # +) e_layers, d_model, d_ff, n_heads, top_k, down_sampling_method, down_sampling_layers, down_sampling_window, ...
    
    # FEDformer
    # +) e_layers, d_model, d_ff, n_heads, moving_avg

    # ETSformer
    # +) e_layers, d_model, d_ff, n_heads, top_k

    # Crossformer
    parser.add_argument('--seg_len_cf', type=int, default=12, help='the length of segment for Crossformer')
    # +) e_layers, d_model, d_ff, n_heads, factor

    # PatchTST
    parser.add_argument('--patch_size', type=int, default=16, help='the patch size')
    parser.add_argument('--patch_stride', type=int, default=8, help='the patch stride')
    # +) e_layers, d_model, d_ff, n_heads

    # GPT4TS (One-fits-all)
    parser.add_argument('--huggingface_cache_dir', type=str, default='./huggingface', help='huggingface cache directory for GPT2')
    # +) d_model, e_layer, d_ff, patch_size, patch_stride

    # iTransformer
    # +) e_layer, d_model, d_ff, n_head, factor

    # InterpretGN
    parser.add_argument('--dnn_type', type=str, default='FCN', choices=['FCN', 'Transformer', 'TimesNet', 'PatchTST', 'ResNet'])
    parser.add_argument('--num_shapelet', type=int, default=10, help='number of shapelet')
    parser.add_argument("--lambda_reg", type=float, default=0.1)
    parser.add_argument("--lambda_div", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=1.)
    parser.add_argument("--distance_func", type=str, default='euclidean', choices=['euclidean', 'cosine', 'pearson'])
    parser.add_argument("--memory_efficient", type=str2bool, default=False)
    parser.add_argument("--sbm_cls", type=str, default='linear', choices=['linear', 'bilinear', 'attention'])
    parser.add_argument("--gating_value", type=float, default=None)

    # TSCMamba
    parser.add_argument('--variation',type=int,default=64,help='variation in CWT')
    parser.add_argument('--wavelet_name',type=str,default='morl',help='Wavelet Name')
    parser.add_argument('--rescale_size',type=int,default=64,help='Rescale size after WT')
    parser.add_argument('--no_rocket',type=int,default=0,help='Switch between rocket/FFN features')
    parser.add_argument('--half_rocket',type=int,default=0,help='half_rocket and half mlp features')
    parser.add_argument('--channel_token_mixing',type=int,default=0,help='Mixed features from channel and token for rocket. unmentioned in paper')
    parser.add_argument('--initial_focus',type=float,default=1.0,help='Initial focus parameter')
    parser.add_argument('--additive_fusion',type=int,default=1,help='Either to use additive fusion or multiplicative fusion')
    parser.add_argument('--only_forward_scan',type=int,default=0,help='Only forward scan without reverse scan')
    parser.add_argument('--flip_dir',type=int,default=2,help='Flip Direction')
    parser.add_argument('--reverse_flip',type=int,default=0,help='reverse scan with flipped addition')
    parser.add_argument('--max_pooling',type=int,default=0,help='avg pooling or max pooling')
    # +) d_model(=projected_space), patch_size, e_layers(=num_mambas), d_conv, expand(=e_fact), d_ff(=d_state), 



    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--itr_start', type=int, default=0, help='starting index of experiment times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--use_pretrained', type=str2bool, default=False, help='use pretrained model')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=False,
                        help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",
                        help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",
                        help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",
                        help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",
                        help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    # Medformer
    parser.add_argument('--swa', type=str2bool, default=False, help='whether to use stochastic weight averaging')

    args = parser.parse_args()
    # declare CUDA_VISIBLE_DEVICES before using torch.cuda
    if args.use_gpu and args.gpu_type == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(
            args.gpu) if not args.use_multi_gpu else args.devices
    
    import torch
    import torch.backends
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    from exp.exp_imputation import Exp_Imputation
    from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
    from exp.exp_anomaly_detection import Exp_Anomaly_Detection
    from exp.exp_classification import Exp_Classification

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    if torch.cuda.is_available() and args.use_gpu and args.gpu_type == 'cuda':
        if args.use_multi_gpu:  # multi-gpu
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_indices = [int(id_) for id_ in device_ids]  # e.g. '1,2' -> [1, 2]
            args.device_ids = list(range(len(args.device_indices))) # e.g. [1, 2] -> [0, 1] because of visible devices
            args.gpu = args.device_indices[0]
            args.device = torch.device('cuda:0')
        else:  # one gpu
            args.device = torch.device('cuda')
        print('Using GPU')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() \
        and args.use_gpu and args.gpu_type == 'mps':
        args.device = torch.device("mps")
    else:
        args.device = torch.device("cpu")
        print('Using cpu or mps')

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'classification':
        Exp = Exp_Classification
    elif args.task_name == 'classification_trainlossonly':
        Exp = Exp_Classification_Using_TrainLoss
        args.task_name = 'classification'  # change task name for further process
    elif args.task_name == 'classification_medical':
        Exp = Exp_Classification_Medical
        args.task_name = 'classification'  # change task name for further process
    # elif args.task_name == 'long_term_forecast':
    #     Exp = Exp_Long_Term_Forecast
    # elif args.task_name == 'short_term_forecast':
    #     Exp = Exp_Short_Term_Forecast
    # elif args.task_name == 'imputation':
    #     Exp = Exp_Imputation
    # elif args.task_name == 'anomaly_detection':
    #     Exp = Exp_Anomaly_Detection
    else:
        raise ValueError('This repository is basically for time series classification tasks.')

    if args.is_training:
        for ii in range(args.itr_start, args.itr + args.itr_start):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            if args.model == 'MambaSingleLayer':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}'
            elif args.model == 'MambaMultiLayer':  # for ablation
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_el{args.e_layers}_dm{args.d_model}_ds{args.d_ff}' \
                        + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                        + f'_tvdt{int(args.tv_dt)}_tvB{int(args.tv_B)}_tvC{int(args.tv_C)}_useD{int(args.use_D)}_{args.des}_{ii}'
            elif args.model == 'Mamba':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_el{args.e_layers}_dm{args.d_model}_ds{args.d_ff}_expand{args.expand}_dc{args.d_conv}_{args.des}_{ii}'
            elif args.model == 'DLinear':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_ma{args.moving_avg}_i{args.individual}_{args.des}_{ii}'
            elif args.model == 'LightTS':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_dm{args.d_model}_cs{args.chunk_size}_{args.des}_{ii}'
            elif args.model == 'MTSMixer':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_el{args.e_layers}_dm{args.d_model}' \
                        + f'_fT{int(args.fac_T)}_w{args.down_sampling_window}_fC{int(args.fac_C)}_df{args.d_ff}' \
                        + f'_rev{int(args.use_revin)}_n{args.use_norm}_i{int(args.individual)}_{args.des}_{ii}'
            elif args.model == 'TimesNet':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_el{args.e_layers}_dm{args.d_model}_df{args.d_ff}_nk{args.num_kernels}_tk{args.top_k}' \
                        + f'_{args.des}_{ii}'
            elif args.model == 'ModernTCN':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dim{'-'.join([str(i) for i in args.dims])}' \
                        + f'_nb{'-'.join([str(i) for i in args.num_blocks])}_lk{'-'.join([str(i) for i in args.large_size])}_sk{'-'.join([str(i) for i in args.small_size])}' \
                        + f'_ffr{args.ffn_ratio}_ps{args.patch_size}_str{args.patch_stride}_multi{args.use_multi_scale}' \
                        + f'_merged{args.small_kernel_merged}_{args.des}_{ii}'
            elif args.model == 'TimeMixerPP':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_el{args.e_layers}_dm{args.d_model}_nh{args.n_heads}_df{args.d_ff}' \
                        + f'_{args.down_sampling_method}{args.down_sampling_layers}-{args.down_sampling_window}' \
                        + f'_cm{args.channel_mixing}_ci{args.channel_independence}_oa{args.output_attention}_nk{args.num_kernels}_tk{args.top_k}' \
                        + f'_{args.des}_{ii}'
            elif args.model == 'ETSformer':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}' \
                        + f'_tk{args.top_k}_{args.des}_{ii}'
            elif args.model == 'FEDformer':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}' \
                        + f'_ma{args.moving_avg}_{args.des}_{ii}'
            elif args.model == 'Crossformer':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_el{args.e_layers}_dm{args.d_model}_nh{args.n_heads}_df{args.d_ff}' \
                        + f'_fac{args.factor}_seg{args.seg_len_cf}_{args.des}_{ii}'
            elif args.model == 'PatchTST':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_el{args.e_layers}_dm{args.d_model}_nh{args.n_heads}_df{args.d_ff}' \
                        + f'_ps{args.patch_size}_str{args.patch_stride}_{args.des}_{ii}'
            elif args.model == 'GPT4TS':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_el{args.e_layers}_dm{args.d_model}_df{args.d_ff}' \
                        + f'_ps{args.patch_size}_str{args.patch_stride}_{args.des}_{ii}'
            elif args.model == 'iTransformer':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_el{args.e_layers}_dm{args.d_model}_nh{args.n_heads}_df{args.d_ff}' \
                        + f'_fac{args.factor}_{args.des}_{ii}'
            elif args.model == 'InterpretGN':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_dnn{args.dnn_type}_ns{args.num_shapelet}_div{args.lambda_div}_reg{args.lambda_reg}' \
                        + f'_eps{args.epsilon}_dfunc{args.distance_func}_mem{args.memory_efficient}_cls{args.sbm_cls}_gate{args.gating_value}' \
                        + f'_{args.des}_{ii}'
            elif args.model == 'TSCMamba':
                setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                        + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                        + f'_dm{args.d_model}_cwt{args.variation}_rescale{args.rescale_size}_ps{args.patch_size}' \
                        + f'_el{args.e_layers}_df{args.d_ff}_dconv{args.d_conv}_expand{args.expand}' \
                        + f'_nrocket{args.no_rocket}_hrocket{args.half_rocket}_focus{args.initial_focus}_additive{args.additive_fusion}' \
                        + f'_onlyfs{args.only_forward_scan}_flipdir{args.flip_dir}_reverseflip{args.reverse_flip}' \
                        + f'_maxpool{args.max_pooling}_{args.des}_{ii}'

            else:
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.expand,
                    args.d_conv,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des, ii)

            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)  # set experiments
        ii = 0
        if args.model == 'MambaSingleLayer':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{ii}'
        elif args.model == 'MambaMultiLayer':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_el{args.e_layers}_dm{args.d_model}_ds{args.d_ff}' \
                    + f'_expand{args.expand}_dc{args.d_conv}_nk{args.num_kernels}' \
                    + f'_tvdt{args.tv_dt}_tvB{args.tv_B}_tvC{args.tv_C}_useD{int(args.use_D)}_{args.des}_{ii}'
        elif args.model == 'Mamba':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_el{args.e_layers}_dm{args.d_model}_ds{args.d_ff}_expand{args.expand}_dc{args.d_conv}_{args.des}_{ii}'
        elif args.model == 'DLinear':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_ma{args.moving_avg}_i{args.individual}_{args.des}_{ii}'
        elif args.model == 'LightTS':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_dm{args.d_model}_cs{args.chunk_size}_{args.des}_{ii}'
        elif args.model == 'MTSMixer':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_el{args.e_layers}_dm{args.d_model}' \
                    + f'_fT{int(args.fac_T)}_w{args.down_sampling_window}_fC{int(args.fac_C)}_df{args.d_ff}' \
                    + f'_rev{int(args.use_revin)}_n{args.use_norm}_i{int(args.individual)}_{args.des}_{ii}'
        elif args.model == 'TimesNet':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_el{args.e_layers}_dm{args.d_model}_df{args.d_ff}_nk{args.num_kernels}_tk{args.top_k}' \
                    + f'_{args.des}_{ii}'
        elif args.model == 'ModernTCN':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dim{'-'.join([str(i) for i in args.dims])}' \
                    + f'_nb{'-'.join([str(i) for i in args.num_blocks])}_lk{'-'.join([str(i) for i in args.large_size])}_sk{'-'.join([str(i) for i in args.small_size])}' \
                    + f'_ffr{args.ffn_ratio}_ps{args.patch_size}_str{args.patch_stride}_multi{args.use_multi_scale}' \
                    + f'_merged{args.small_kernel_merged}_{args.des}_{ii}'
        elif args.model == 'TimeMixerPP':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_el{args.e_layers}_dm{args.d_model}_nh{args.n_heads}_df{args.d_ff}' \
                    + f'_{args.down_sampling_method}{args.down_sampling_layers}-{args.down_sampling_window}' \
                    + f'_cm{args.channel_mixing}_ci{args.channel_independence}_oa{args.output_attention}_nk{args.num_kernels}_tk{args.top_k}' \
                    + f'_{args.des}_{ii}'
        elif args.model == 'ETSformer':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}' \
                    + f'_tk{args.top_k}_{args.des}_{ii}'
        elif args.model == 'FEDformer':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}' \
                    + f'_ma{args.moving_avg}_{args.des}_{ii}'
        elif args.model == 'Crossformer':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_el{args.e_layers}_dm{args.d_model}_nh{args.n_heads}_df{args.d_ff}' \
                    + f'_fac{args.factor}_seg{args.seg_len_cf}_{args.des}_{ii}'
        elif args.model == 'PatchTST':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_el{args.e_layers}_dm{args.d_model}_nh{args.n_heads}_df{args.d_ff}' \
                    + f'_ps{args.patch_size}_str{args.patch_stride}_{args.des}_{ii}'
        elif args.model == 'GPT4TS':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_el{args.e_layers}_dm{args.d_model}_df{args.d_ff}' \
                    + f'_ps{args.patch_size}_str{args.patch_stride}_{args.des}_{ii}'
        elif args.model == 'iTransformer':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_el{args.e_layers}_dm{args.d_model}_nh{args.n_heads}_df{args.d_ff}' \
                    + f'_fac{args.factor}_{args.des}_{ii}'
        elif args.model == 'InterpretGN':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_dnn{args.dnn_type}_ns{args.num_shapelet}_div{args.lambda_div}_reg{args.lambda_reg}' \
                    + f'_eps{args.epsilon}_dfunc{args.distance_func}_mem{args.memory_efficient}_cls{args.sbm_cls}_gate{args.gating_value}' \
                    + f'_{args.des}_{ii}'
        elif args.model == 'TSCMamba':
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_ft{args.features}' \
                    + f'_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}' \
                    + f'_dm{args.d_model}_cwt{args.variation}_rescale{args.rescale_size}_ps{args.patch_size}' \
                    + f'_el{args.e_layers}_df{args.d_ff}_dconv{args.d_conv}_expand{args.expand}' \
                    + f'_nrocket{args.no_rocket}_hrocket{args.half_rocket}_focus{args.initial_focus}_additive{args.additive_fusion}' \
                    + f'_onlyfs{args.only_forward_scan}_flipdir{args.flip_dir}_reverseflip{args.reverse_flip}' \
                    + f'_maxpool{args.max_pooling}_{args.des}_{ii}'
        else:
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.expand,
                args.d_conv,
                args.factor,
                args.embed,
                args.distil,
                args.des, ii)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        if args.gpu_type == 'mps':
            torch.backends.mps.empty_cache()
        elif args.gpu_type == 'cuda':
            torch.cuda.empty_cache()
