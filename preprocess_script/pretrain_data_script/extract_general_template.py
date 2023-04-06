'''
使用templatecorr的环境来抽取通用的模板
ref: https://github.com/hesther/templatecorr
'''


import pandas as pd
import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rdchiral.template_extractor import extract_from_reaction
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
from pandarallel import pandarallel


def get_templates(rxn_smi, 
                  prec, 
                  no_special_groups=True, 
                  radius=0,      # 抽取最通用的模板
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
        template = extract_from_reaction(reaction,no_special_groups=no_special_groups,radius=radius)["reaction_smarts"]
        if add_brackets:
            template = "(" + template.replace(">>", ")>>")
    except:
        template = None  
    #Validate:
    if template != None:
        rct = rdchiralReactants(rxn_smi.split(">")[-1])
        try:
            rxn = rdchiralReaction(template)
            outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
        except:
            outcomes =[]
        if not prec in outcomes:
            template=None
    return template


if __name__ == '__main__':
    pandarallel.initialize(nb_workers=10, progress_bar=True)
    pretrain_data_path = '../../dataset/pretrain_data'

    database = pd.read_csv(
        '../../dataset/source_dataset/USPTO_remapped_remove_same_rxn_templates.csv')
    
    database['precursors'] = database['clean_map_rxn'].parallel_apply(lambda x:x.split('>>')[0])
    database['retro_template_r0'] = database.parallel_apply(lambda row:get_templates(row['droped_unmapped_rxn'], row['precursors'], no_special_groups=True, radius=0), axis=1)
    print('extract erro # {}'.format(database['retro_template_r0'].isna().sum())) 

    database.to_csv('../../dataset/source_dataset/USPTO_remapped_remove_same_rxn_templates_template_r0.csv', index=False)
