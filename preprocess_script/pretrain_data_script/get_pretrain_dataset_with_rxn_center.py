import json
import os
import signal
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import numpy as np
from rxnfp.tokenization import RegexTokenizer
import sys
sys.path.append('../../')
from models.utils import identify_attention_token_idx_for_rxn_component
import torch
from pandarallel import pandarallel

'''
取出clean_map_rxn用于masked lm模型训练 使用反应中心策略
'''
class timeout:
    """
    Function for counting time. If a process takes too long to finish, it will be exited by this function.
    """

    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds

        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

def mark_rxn_center(smiles, sub_smarts):

    try:
        mol = Chem.MolFromSmiles(smiles)
        patt = Chem.MolFromSmarts(sub_smarts)
        with timeout(seconds=100000):
            match_idx = mol.GetSubstructMatch(patt)
            if match_idx:
                for idx, atom in enumerate(mol.GetAtoms()):
                    if idx in match_idx:
                        atom.SetProp('molAtomMapNumber', '0')
    except Exception as e:
        print(smiles)
        return smiles
    return Chem.MolToSmiles(mol, canonical=False)

def get_reaction_center(smiles, sub_smarts):

    try:
        mol = Chem.MolFromSmiles(smiles)
        patt = Chem.MolFromSmarts(sub_smarts)
        with timeout(seconds=100000):
            match_idx = mol.GetSubstructMatch(patt)
            return match_idx
    except Exception as e:

        return ()
    
def get_template_label_dict(template_series: pd.Series, threshold=5):
    template_list = list(zip(template_series.value_counts().index,
                        template_series.value_counts(),
                        template_series.value_counts(normalize=True)
                       )
                   )
    template_columns = ['retro_template', 'count', 'frequency']
    template_df = pd.DataFrame(template_list, columns=template_columns)
    template_df['cumulative'] = template_df.frequency.cumsum()

    template_df['template_labels'] =  template_df.index
    template_df.loc[(template_df['count'] <= threshold), ['template_labels']] = -100   # 少于阈值的模板不算损失

    template_to_lables = {row['retro_template']:row['template_labels'] for index, row in tqdm(template_df.iterrows(),total=len(template_df))}

    return template_to_lables, template_df.loc[(template_df['count'] <= threshold)]['count'].sum()

def template_to_labels_function(template):
    
    try:
        return template_to_lables[template]
    except:
        return -100







def get_rxn_center_mask_label(rxn, retro_tpl, tokenizer):
    rxn_tokens = tokenizer.tokenize(rxn)

    _reactants_token_idx, _product_token_idx, atom_token_mask = identify_attention_token_idx_for_rxn_component(rxn_tokens)
    reaction_center_token_mask = torch.zeros_like(atom_token_mask).bool()
    if pd.isna(retro_tpl) or (len(retro_tpl) >= 700):

        return reaction_center_token_mask.tolist()
    
    react, prod = rxn.split('>>')
    sub_prod, sub_react = retro_tpl.split('>>')
    react_center = get_reaction_center(react, sub_react)
    prod_center = get_reaction_center(prod, sub_prod)
    if react_center:
        react_center_idx = _reactants_token_idx[torch.tensor(react_center)]
    else:
        react_center_idx = torch.tensor([])
    if prod_center:
        prod_center_idx = _product_token_idx[torch.tensor(prod_center)]
    else:
        prod_center_idx = torch.tensor([])
    all_center_idx = torch.cat([react_center_idx, prod_center_idx], dim=-1).int()
    if all_center_idx.equal(torch.tensor([]).int()): 
        return reaction_center_token_mask.tolist()

    reaction_center_token_mask[all_center_idx.long()] = True
    
    return reaction_center_token_mask.tolist()

def replace_function(x):
    if pd.isna(x):
        return x
    else:
        x = x[1:].replace(')>>', '>>')
        return x.replace('&', ';')
        

if __name__ == '__main__':
    
    # 逻辑： mark rxn center 的反应smiles和无标记的反应smiles在分词之后token不同的部分要着重关注
    pandarallel.initialize(nb_workers=10, progress_bar=True)
    tqdm.pandas(desc='pandas bar')

    threshold=5

    pretrain_data_path = '../../dataset/pretrain_data'
    basic_tokenizer = RegexTokenizer()

    if not os.path.exists(pretrain_data_path):
        os.makedirs(pretrain_data_path)
    database = pd.read_csv(
        '../../dataset/source_dataset/USPTO_remapped_remove_same_rxn_templates_template_r0.csv')
    
    database['retro_template_r0'] = database['retro_template_r0'].apply(lambda x:replace_function(x))
    template_to_lables, masked_cnt = get_template_label_dict(database['retro_template_r0'], threshold=threshold)
    masked_cnt += database['retro_template_r0'].isna().sum()
    cls_number = len(set(list(template_to_lables.values())))
    print('template labels class number:', cls_number)
    print('template masked count:', masked_cnt)
    with open(os.path.join(pretrain_data_path, 'rxn_center_modeling_template_to_lables_thr_{}_cls_num_{}.json'.format(threshold, cls_number)), 'w', encoding='utf-8') as f:

        json.dump(template_to_lables, f)

    database['template_labels'] = database['retro_template_r0'].parallel_apply(lambda x:template_to_labels_function(x))

    database['all_reaction_center_index_mask'] = database.parallel_apply(lambda row: get_rxn_center_mask_label(row['clean_map_rxn'], row['retro_template_r0'], tokenizer=basic_tokenizer), axis=1)
    canonicalize_rxns = database['clean_map_rxn']
    canonicalize_rxn_val = canonicalize_rxns.sample(1000, random_state=123)
    canonicalize_rxn_train = canonicalize_rxns.drop(canonicalize_rxn_val.index)

    database.loc[canonicalize_rxn_train.index, 'dataset'] = 'train'
    database.loc[canonicalize_rxn_val.index, 'dataset'] = 'val'
    database = database[['clean_map_rxn', 'all_reaction_center_index_mask', 'template_labels', 'dataset']]
    
    database.to_pickle(os.path.join(pretrain_data_path, 'rxn_center_modeling.pkl'))
    
    debug_database = database.sample(2000, random_state=123)
    debug_database.to_pickle(os.path.join(pretrain_data_path, 'rxn_center_modeling_debug.pkl'))


