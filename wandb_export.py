import os
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

#print(results)

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

        #print(results[group][jt])
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
runs_df.to_csv("Results_Tables/All.csv")


datasets = set(df_out['dataset'])   # Get unique dataset values

for dataset in datasets:
    df_out = {}
    for stat_key in stats.keys():
        data = []
        for group_key in sorted(results):
            for jt_key in sorted(results[group_key]):
                result = results[group_key][jt_key]
                if result != {} and result['dataset'] == dataset and result['scale'] == 2:
                    data.append(results[group_key][jt_key][stat_key])
                    #print(data)
        df_out[stat_key] = data
    #print()
    #print(df_out)
    ds_df = pd.DataFrame(df_out)
    ds_df.to_csv('Results_Tables/'+dataset+".csv")
