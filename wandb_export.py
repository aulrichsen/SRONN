import os
import math
from platform import mac_ver
import pandas as pd 
import wandb
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("au-phd/HSI-Super-Resolution")

decision_metric = 'test_SSIM_from_best_SSIM_model'  # Metric to determine which statistics to keep

results = {}

for run in runs: 
    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config = {k: v for k,v in run.config.items()
          if not k.startswith('_')}

    group = run.group
    if group not in results.keys(): results[group] = {}
    
    jt = config['wandb_jt']
    if jt not in results[group].keys(): results[group][jt] = {}

for run in runs: 
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files 
    summary = run.summary._json_dict

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config = {k: v for k,v in run.config.items()
          if not k.startswith('_')}

    group = run.group
    if group not in results: results.append(group)
    jt = config['wandb_jt']

    if decision_metric in summary.keys():
        # If run finished training

        stats = {'model': config['model'], 'is_residual': config['is_residual'], 'norm': config['norm_type'], 'num_params': config['num_params'], 'dataset': config['dataset'], 'scale': config['scale'], 'lr': config['lr'], 'lr_milestones': config['lr_milestones'], 'PSNR': summary['test_PSNR_from_best_SSIM_model'], 'SSIM': summary['test_SSIM_from_best_SSIM_model'], 'SAM': summary['test_SAM_from_best_SSIM_model']}

        if results[group][jt] == {}:
            # If no previous results recorded
            results[group][jt] = stats
        elif summary[decision_metric] > results[group][jt]['SSIM']:
            # If results better than results already recorded            
            results[group][jt] = stats

# Sort
df_out = {}
for stat_key in stats.keys():
    data = []
    for group_key in sorted(results):
        for jt_key in sorted(results[group_key]):
            if results[group_key][jt_key] != {}:
                data.append(results[group_key][jt_key][stat_key])
    df_out[stat_key] = data

if not os.path.isdir('Results_Tables'):
    os.mkdir('Results_Tables')

runs_df = pd.DataFrame(df_out)
runs_df = runs_df.sort_values(['model', 'is_residual', 'norm'])
runs_df.to_csv("Results_Tables/All.csv", index=False)


datasets = set(df_out['dataset'])   # Get unique dataset values

# Rename wandb filed names to titles to be used in paper
stat_key_map = {'model': 'Model', 
                'is_residual': 'Residual', 
                'norm': 'Normalization', 
                'lr': 'lr', 
                'num_params': '\# parameters', 
                'dataset': 'Dataset',
                'lr_milestones': 'lr steps', 
                'PSNR': 'PSNR', 
                'SSIM': 'SSIM', 
                'SAM': 'SAM'}

# ======================
# Res and non Res Results (No norm)
# ======================

stat_key_ignore = ['scale', 'is_residual', 'norm']      # Don't bother recording scale

for is_res in [False, True]:
    df_out = {}
    for stat_key in stats.keys():
        if stat_key not in stat_key_ignore:
            data = []
            for group_key in sorted(results):
                for jt_key in sorted(results[group_key]):
                    result = results[group_key][jt_key]
                    if result != {} and result['is_residual'] == is_res and result['norm'] == 'none' and result['scale'] == 2 and result['dataset'] != 'KSC':
                        values = results[group_key][jt_key][stat_key]

                        # Format specific stats
                        if stat_key == 'lr_milestones':
                            # Convert from [5000, 40000] to 5k, 40k
                            formatted_values = ""
                            for value in values:
                                value = value/1000
                                if (value - int(value)) == 0:
                                    value = int(value)      # Convert to int to remove .0
                                formatted_values += str(value) + 'k, '     # Need to round!
                            formatted_values = formatted_values[:-2]    # Remove final ', '
                        elif stat_key == 'lr':
                            formatted_values = '$1 \\times 10^{'+str(round(math.log(values, 10)))+'}$'
                        elif stat_key == 'PSNR' or stat_key == 'SAM':
                            formatted_values = round(values, 3)
                        elif stat_key == 'SSIM':
                            formatted_values = round(values, 4)    
                        elif stat_key == 'dataset' and values == 'PaviaU':
                            formatted_values = 'Pavia \par University'
                        else:
                            formatted_values = values   # Keep same

                        if values == 'SRONN_AEP': formatted_values = 'sSRONN'

                        data.append(formatted_values)
                        #print(data)
            df_out[stat_key_map[stat_key]] = data
    
    res_msg = '_residual' if is_res else ''

    ds_df = pd.DataFrame(df_out)
    col_list = list(ds_df)
    col_list[0], col_list[1], col_list[2] = col_list[2], col_list[0], col_list[1]   # model, num_params, dataset -> dataset, model num_params
    ds_df = ds_df[col_list]
    ds_df = ds_df.sort_values(['Dataset', 'Model'])
    ds_df = ds_df.reset_index(drop=True)

    # Find best values for each dataset
    max_idxs, min_idxs = [], []
    for i in range(0, ds_df.shape[0], 3):
        max_idxs.append(ds_df[i:i+3].idxmax(axis=0, numeric_only=True))
        min_idxs.append(ds_df[i:i+3].idxmin(axis=0, numeric_only=True))

    # Make best values latex bold
    for max_idx, min_idx in zip(max_idxs, min_idxs):
        ds_df.iloc[max_idx['PSNR'], 5] = '\\textbf{'+str(ds_df.iloc[max_idx['PSNR'], 5])+'}'
        ds_df.iloc[max_idx['SSIM'], 6] = '\\textbf{'+str(ds_df.iloc[max_idx['SSIM'], 6])+'}'
        ds_df.iloc[min_idx['SAM'], 7] = '\\textbf{'+str(ds_df.iloc[min_idx['SAM'], 7])+'}'
        ds_df.iloc[min_idx['\# parameters'], 2] = '\\textbf{'+str(ds_df.iloc[min_idx['\# parameters'], 2])+'}'
    ds_df.to_csv('Results_Tables/no_norm'+res_msg+".csv", index=False)


# ======================
# Normalization Results
# ======================

stat_key_ignore = ['scale', 'dataset']      # Don't bother recording scale or dataset (same for all)

for dataset in datasets:
    df_out = {}
    for stat_key in stats.keys():
        if stat_key not in stat_key_ignore:
            data = []
            for group_key in sorted(results):
                for jt_key in sorted(results[group_key]):
                    result = results[group_key][jt_key]
                    if result != {} and result['dataset'] == dataset and result['scale'] == 2:
                        values = results[group_key][jt_key][stat_key]

                        # Format specific stats
                        if stat_key == 'lr_milestones':
                            # Convert from [5000, 40000] to 5k, 40k
                            formatted_values = ""
                            for value in values:
                                value = value/1000
                                if (value - int(value)) == 0:
                                    value = int(value)      # Convert to int to remove .0
                                formatted_values += str(value) + 'k, '     # Need to round!
                            formatted_values = formatted_values[:-2]    # Remove final ', '
                        elif stat_key == 'is_residual':
                            formatted_values = 'yes' if values else 'no'
                        elif stat_key == 'lr':
                            formatted_values = '$1 \\times 10^{'+str(round(math.log(values, 10)))+'}$'
                        elif stat_key == 'PSNR' or stat_key == 'SAM':
                            formatted_values = round(values, 3)
                        elif stat_key == 'SSIM':
                            formatted_values = round(values, 4)
                        else:
                            formatted_values = values   # Keep same

                        if values == 'SRONN_AEP': formatted_values = 'sSRONN'

                        data.append(formatted_values)

            df_out[stat_key_map[stat_key]] = data

    ds_df = pd.DataFrame(df_out)
    ds_df = ds_df.sort_values(['Model', 'Residual', 'Normalization'])
    ds_df = ds_df.reset_index(drop=True)
    
    # Find best values for each dataset
    max_idx = ds_df.idxmax(axis=0, numeric_only=True)
    min_idx = ds_df.idxmin(axis=0, numeric_only=True)

    max_vals = ds_df.max(axis=0, numeric_only=True)
    min_vals = ds_df.min(axis=0, numeric_only=True)

    min_idxs = ds_df.index[ds_df['\# parameters']==ds_df['\# parameters'].min()].tolist()   # Get multiple indexes for number of parameters

    # Make best values latex bold
    ds_df.iloc[max_idx['PSNR'], 6] = '\\textbf{'+str(ds_df.iloc[max_idx['PSNR'], 6])+'}'
    ds_df.iloc[max_idx['SSIM'], 7] = '\\textbf{'+str(ds_df.iloc[max_idx['SSIM'], 7])+'}'
    ds_df.iloc[min_idx['SAM'], 8] = '\\textbf{'+str(ds_df.iloc[min_idx['SAM'], 8])+'}'
    #ds_df.iloc[min_idxs['\# parameters'], 3] = '\\texbf{'+str(ds_df.iloc[min_idxs['\# parameters'], 3])+'}'
    ds_df.iloc[min_idxs, 3] = '\\textbf{'+str(ds_df.iloc[min_idx['\# parameters'], 3])+'}'

    ds_df.to_csv('Results_Tables/'+dataset+".csv", index=False)

