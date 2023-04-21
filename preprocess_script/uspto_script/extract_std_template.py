'''
使用templatecorr的环境来抽取通用的模板
ref: https://github.com/hesther/templatecorr
'''


import os
import pandas as pd
import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdchiral.template_extractor import extract_from_reaction
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
from pandarallel import pandarallel


def get_templates(rxn_smi, 
                  prec,     
                  add_brackets=True):
    """
    Adapt from https://github.com/hesther/templatecorr/blob/main/templatecorr/extract_templates.py#L22

    Extracts a template at a specified level of specificity for a reaction smiles.

    :param rxn_smi: Reaction smiles string
    :param prec: Canonical smiles string of precursor
    :param no_special_groups: Boolean whether to omit special groups in template extraction
    :param radius: Integer at which radius to extract templates
    :param add_brackets: Whether to add brackets to make template pseudo-unimolecular

    :return: Template
    """    
    #Extract:
    try:
        rxn_split = rxn_smi.split(">")
        reaction={"_id":0,"reactants":rxn_split[0],"spectator":rxn_split[1],"products":rxn_split[2]}
        template = extract_from_reaction(reaction)["reaction_smarts"]
        if add_brackets:
            template = "(" + template.replace(">>", ")>>")
    except:
        template = ''  
    #Validate:
    if template != '':
        rct = rdchiralReactants(rxn_smi.split(">")[-1])
        try:
            rxn = rdchiralReaction(template)
            outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
        except:
            outcomes =[]
        if not prec in outcomes:
            template=''
    return template


if __name__ == '__main__':
    pandarallel.initialize(nb_workers=10, progress_bar=True)
    source_dataset_path = '../../dataset/source_dataset/'

    uspto_condition_df = pd.read_csv(os.path.join(source_dataset_path, 'USPTO_condition_final', 'USPTO_condition_pred_category_maped_rxn.csv'))
    
    uspto_condition_df['precursors'] = uspto_condition_df['canonical_rxn'].parallel_apply(lambda x:x.split('>>')[0])
    uspto_condition_df['retro_template'] = uspto_condition_df.parallel_apply(lambda row:get_templates(row['remapped_rxn'], row['precursors'], add_brackets=False), axis=1)
    print('extract erro # {}'.format((uspto_condition_df['retro_template']=='').sum())) 

    uspto_condition_df.to_csv(os.path.join(source_dataset_path, 'USPTO_condition_final', 'USPTO_condition_pred_category_maped_rxn_tpl.csv'), index=False)
