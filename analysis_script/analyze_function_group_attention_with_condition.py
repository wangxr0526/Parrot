# %%
import sys
sys.path.append('..')
from models.parrot_model import ParrotConditionPredictionModel
from IPython.core.display import display, HTML, Javascript, SVG
import yaml
import os
import torch
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from pandarallel import pandarallel
from models.utils import load_dataset, print_args, inference_load
import seaborn as sns


# %%
config_file = '../configs/config_inference_use_uspto.yaml'
subgraph_file = './eval_data/eval_use_data/frag_cnt_nubmer_10000.txt'
input_path='../dataset/source_dataset/USPTO_condition_final/USPTO_condition_pred_category_maped_rxn_tpl.csv'      # 相较于原数据集添加了反应类别名称
analysis_results_save_path = './eval_data/group_attn_results'

debug = True

# %%

config = yaml.load(open(config_file, "r"),
                    Loader=yaml.FullLoader)


model_args = config['model_args']

model_args.update(
    {
    'pretrained_path': os.path.join('..', model_args['pretrained_path']),
    'output_dir': os.path.join('..', model_args['output_dir']),
    'best_model_dir': os.path.join('..', model_args['best_model_dir']),
    'output_attention': True,
    }
)
dataset_args = config['dataset_args']

dataset_args.update(
    {
    'dataset_root': os.path.join('..', dataset_args['dataset_root'])
    })

print(
    '\n########################\Attention Evaluation configs:\n########################\n'
)
print(yaml.dump(config))
print('\n########################\n')


dataset_df, condition_label_mapping = load_dataset(**dataset_args)
dataset_df_with_rxn_class = pd.read_csv(os.path.join(dataset_args['dataset_root'], dataset_args['database_fname'].replace('.csv', '_pred_category_maped_rxn_tpl.csv')))

model_args['use_temperature'] = dataset_args['use_temperature']
model_args['decoder_args'].update({
    'tgt_vocab_size': len(condition_label_mapping[0]),
    'condition_label_mapping': condition_label_mapping
    })







model = ParrotConditionPredictionModel("bert", model_args['best_model_dir'], args=model_args, use_cuda=torch.cuda.is_available())


test_df = dataset_df.loc[dataset_df['dataset'] == 'test'].reset_index(drop=True)
if debug:
    test_df = test_df[:1000]
test_df = test_df[['canonical_rxn', 'condition_labels']]
test_df.columns = ['text', 'labels']
score_map_condition_type_dict = model.analyze_function_group_attention_with_condition(test_df, test_batch_size=16, subgraph_fpath=subgraph_file, analysis_results_save_path=analysis_results_save_path)


# %%


# %%
import seaborn as sns
condition_types = ['c1', 's1', 's2', 'r1', 'r2']
for c_type in condition_types:
    print(f'#######################################  {c_type}  ####################################')
    score_map_df = score_map_condition_type_dict[c_type]

#     score_map_df = score_map_df.style.set_properties(**{
#     # 'background-color': 'Reds',
#     'background_gradient': 'Reds',
#     'font-size': '20pt',
# })
    score_map_df = score_map_df.style.background_gradient(cmap='Reds')
    score_map_df.to_excel(os.path.join(analysis_results_save_path, '{}_score_map.xlsx'.format(c_type)))
    # score_map_df = score_map_df.style.set_table_attributes('style="font-size: 25px"')
    display(score_map_df)

    # sns.heatmap(score_map_df)

# %%
draw_catalyst_list = [
    'c1ccc([P](c2ccccc2)(c2ccccc2)[Pd]([P](c2ccccc2)(c2ccccc2)c2ccccc2)([P](c2ccccc2)(c2ccccc2)c2ccccc2)[P](c2ccccc2)(c2ccccc2)c2ccccc2)cc1',
    '[Ru]',
    '[Pt]',
    'O=[Os](=O)(=O)=O',
    ]

# %%
from matplotlib import pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.image as image
from rdkit.Chem import Draw
from matplotlib.patches import FancyBboxPatch

def draw_mol_to_png(smi, figname):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        Draw.MolToFile(
            mol,  # mol对象
            figname,  # 图片存储地址
            size=(300, 300), 
            kekulize=True, 
            wedgeBonds=True, 
            imageType=None, 
            fitImage=False, 
            options=None, 
            # **kwargs
        )

def draw_score_for_one_condition(df, condition_name, remove_zero=True, topn=15, figsize=(30,16), dpi=600, score_bar_save_path=None):
    sns.set(context='notebook', style='ticks', font_scale=1.5)
    if not remove_zero:
        function_group_score_series = df[condition_name]
    else:
        function_group_score_series = df[condition_name][df[condition_name] != 0.0]
    
    function_group_score_with_name = list(zip(function_group_score_series.index.tolist(), function_group_score_series.tolist()))
    function_group_score_with_name.sort(key=lambda x:x[1], reverse=True)
    function_group_score_with_name = function_group_score_with_name[:topn]
    figure, axes = plt.subplots(nrows=1, ncols=1, figsize=figsize, dpi=dpi)

    fig = sns.barplot(
                # x=[x[0] for x in function_group_score_with_name],
                x=[f'h{x}'  for x in range(len(function_group_score_with_name))],
                y=[x[1] for x in function_group_score_with_name], 
                # ax=axes
                # color = '#b9e38d'
                palette=sns.color_palette('Greens_r', 20),
                )
    fig.set_ylim(
        ymin=max(0, (function_group_score_with_name[-1][1] - (function_group_score_with_name[0][1] - function_group_score_with_name[-1][1])*2/3)),
                 )
    fig.set_zorder(0)
#     new_patches = []
#     for patch in reversed(fig.patches):
#         bb = patch.get_bbox()
#         color = patch.get_facecolor()
#         p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
#                                 abs(bb.width), abs(bb.height),
#                                 boxstyle="round,pad=-0.0040,rounding_size=0.015",
#                                 ec="none", fc=color,
#                                 mutation_aspect=4
#                                 )
#         patch.remove()
#         new_patches.append(p_bbox)
#     for patch in new_patches:
#         fig.add_patch(patch)
    # plt.bar([x[0] for x in function_group_score_with_name], 
    #         [x[1] for x in function_group_score_with_name], axes=axes)
    plt.xticks(rotation=60)
    # axes.set_title('Catalyst: {}'.format(condition_name))
    axes.set_title('Catalyst')
    axes.spines['right'].set_color('none')
    axes.spines['top'].set_color('none')
    plt.savefig(score_bar_save_path, bbox_inches='tight')

#     plt.show()
    return function_group_score_with_name

def draw_score_for_one_condition_add_frag(df, condition_name, remove_zero=True, one_group_fig_path=None):
    if not os.path.exists(one_group_fig_path):
        os.makedirs(one_group_fig_path)
    topn = 15
    score_bar_save_path = os.path.join(one_group_fig_path, 'bar.png')
    condition_fig_save_path = os.path.join(one_group_fig_path, 'condition.png')
    
    function_group_score_with_name = draw_score_for_one_condition(df, condition_name, figsize=(30,16), dpi=300, score_bar_save_path=score_bar_save_path)
    
    fragments = [x[0] for x in function_group_score_with_name]
    # fragments = [Chem.MolFromSmiles(smi) for smi in fragments_smiles]
    
    draw_mol_to_png(condition_name, condition_fig_save_path)
    
    [draw_mol_to_png(smi, os.path.join(one_group_fig_path, f'{idx}.png')) for idx, smi in enumerate(fragments)]
    
    fig, ax = plt.subplots(figsize=(32,25), dpi=300)
    
    barxy = [0.5, 0.45]
    bar_img_arr = plt.imread(score_bar_save_path)
    imagebox = OffsetImage(bar_img_arr, zoom=0.22)
    ab = AnnotationBbox(imagebox, barxy,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.0,
                        frameon=False,
                        )
    imagebox.image.axes = ax
    ax.add_artist(ab)
    
    condition_xy = [0.515, 0.68]
    condition_img_arr = plt.imread(condition_fig_save_path)
    imagebox = OffsetImage(condition_img_arr, zoom=0.6)
    imagebox.image.axes = ax

    ab = AnnotationBbox(imagebox, condition_xy,
                        xycoords='data',
                        boxcoords="offset points",
                        pad=0.5,
                        frameon=False,
                        )
    ax.add_artist(ab)
    
    frag_xb = 0.115
    frag_xe = 0.918
    for idx in range(len(fragments)):
        frag_img_arr = plt.imread(os.path.join(one_group_fig_path, f'{idx}.png'))
        frag_xy = [frag_xb + idx*(frag_xe - frag_xb)/(len(fragments) - 1), 0.13]
        imagebox = OffsetImage(frag_img_arr, zoom=0.28)
        imagebox.image.axes = ax

        ab = AnnotationBbox(imagebox, frag_xy,
                            xycoords='data',
                            boxcoords="offset points",
                            pad=0.5,
                            frameon=False,
                            )

        ax.add_artist(ab)
    
    plt.axis('off')
    plt.savefig(os.path.join(one_group_fig_path, 'end_results.png'), bbox_inches='tight')
    

# %%
for idx, condition_name in enumerate(draw_catalyst_list):
    draw_score_for_one_condition_add_frag(score_map_condition_type_dict['c1'], condition_name, one_group_fig_path=f'./figure/conditon_{idx}/')
    # break

# %%



