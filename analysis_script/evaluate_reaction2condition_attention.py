from argparse import ArgumentParser
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from typing import List
from pandarallel import pandarallel
import yaml
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from rdkit import Chem
from models.model_layer import BOS
from models.utils import caonicalize_rxn_smiles, identify_attention_token_idx_for_rxn_component, inference_load

from models.parrot_model import ParrotConditionPredictionModel


def get_report_df(mat, cnt, index, columns):
    report_mat = mat / cnt
    report_df = pd.DataFrame(report_mat, index=index, columns=columns)
    report_df = report_df.round(6)
    return report_df


def calculate_score(pred_idx: set, gt_idx: set, num_atoms: int):
    all_atom_idx = set([i for i in range(num_atoms)])
    TP = len(pred_idx.intersection(gt_idx))
    FP = len(pred_idx.difference(gt_idx))
    TN = len(all_atom_idx.difference((gt_idx).union(pred_idx)))

    if (FP + TP) != len(pred_idx):
        print("there is a mistake in the calculations")

    overlap_score = TP / len(gt_idx) if len(gt_idx) != 0 else 1
    false_positive_rate = FP / (FP + TN) if all_atom_idx != gt_idx else 0

    center_accuracy_two = TP >= 2
    center_accuracy_half = TP >= (len(gt_idx) / 2)

    return overlap_score, false_positive_rate, center_accuracy_two, center_accuracy_half


def get_center_score(attn, center_idx, num_atoms, n_std: float = 0.0):
    center_idx = set(center_idx)

    attn_mean = attn.mean(-1)[:, :, np.newaxis]
    attn_std = attn.std(-1)[:, :, np.newaxis]

    overlap_score_mat = np.zeros(attn.shape[:2])
    false_positive_rate_mat = np.zeros(attn.shape[:2])
    center_accuracy_two_mat = np.zeros(attn.shape[:2])
    center_accuracy_half_mat = np.zeros(attn.shape[:2])
    n_layer, n_head = attn.shape[:2]
    for l in range(n_layer):
        for h in range(n_head):
            sub_attn = attn[l][h]
            sub_mean = attn_mean[l][h]
            sub_std = attn_std[l][h]
            sub_mask = sub_attn > sub_mean + n_std * sub_std
            pred_center_idx = np.argwhere(sub_mask).reshape(-1).tolist()
            pred_center_idx = set(pred_center_idx)

            overlap_score, false_positive_rate, center_accuracy_two, center_accuracy_half = calculate_score(
                pred_center_idx, center_idx, num_atoms)
            overlap_score_mat[l][h] = overlap_score
            false_positive_rate_mat[l][h] = false_positive_rate
            center_accuracy_two_mat[l][h] = center_accuracy_two
            center_accuracy_half_mat[l][h] = center_accuracy_half
    return overlap_score_mat, false_positive_rate_mat, center_accuracy_two_mat, center_accuracy_half_mat


def report_attn_test(preds,
                     attn,
                     tokens,
                     gt,
                     rxn,
                     tpl,
                     condition_type_idx,
                     normalize_react_and_prod: bool,
                     n_std: float = 0.0):

    pred = preds[condition_type_idx]

    # if (pred == gt) and (gt != ''):
    # if (pred == gt):
    # if (gt != ''):
    # if pred != '':
    if True:
        # cnt += 1
        attn_to_eval = attn[:, :, condition_type_idx, :]
        _reactants_token_idx, _product_token_idx, atom_token_mask = [
            x.to(torch.device('cpu')).numpy()
            for x in identify_attention_token_idx_for_rxn_component(
                src_tokens=tokens)
        ]

        react_attn = attn_to_eval[:, :, _reactants_token_idx]
        prod_attn = attn_to_eval[:, :, _product_token_idx]
        if normalize_react_and_prod:
            row_sums_react = react_attn.sum(axis=-1)
            react_attn = np.divide(
                react_attn,
                row_sums_react[:, :, np.newaxis],
                out=np.zeros_like(react_attn),
                where=row_sums_react[:, :, np.newaxis] != 0,
            )
            row_sums_prod = prod_attn.sum(axis=-1)
            prod_attn = np.divide(
                prod_attn,
                row_sums_prod[:, :, np.newaxis],
                out=np.zeros_like(prod_attn),
                where=row_sums_prod[:, :, np.newaxis] != 0,
            )

        react_mol, prod_mol = [
            Chem.MolFromSmiles(smi) for smi in rxn.split('>>')
        ]
        num_react_atoms = react_mol.GetNumAtoms()
        num_prod_atoms = prod_mol.GetNumAtoms()

        prod_subgraph, react_subgraph = [
            Chem.MolFromSmarts(smarts) for smarts in tpl.split('>>')
        ]

        react_match_tpl_idx = react_mol.GetSubstructMatch(react_subgraph)

        react_match_tpl_idx = list(react_match_tpl_idx)
        react_match_tpl_idx.sort()
        react_not_match_tpl_idx = [
            i for i in range(react_mol.GetNumAtoms())
            if i not in react_match_tpl_idx
        ]
        prod_match_tpl_idx = prod_mol.GetSubstructMatch(prod_subgraph)
        prod_match_tpl_idx = list(prod_match_tpl_idx)
        prod_match_tpl_idx.sort()
        prod_not_match_tpl_idx = [
            i for i in range(prod_mol.GetNumAtoms())
            if i not in prod_match_tpl_idx
        ]

        react_ol_mat, react_fpr_mat, react_cent_acc_two_mat, react_cent_acc_half_mat = get_center_score(
            react_attn,
            react_match_tpl_idx,
            num_atoms=num_react_atoms,
            n_std=n_std)
        prod_ol_mat, prod_fpr_mat, prod_cent_acc_two_mat, prod_cent_acc_half_mat = get_center_score(
            prod_attn,
            prod_match_tpl_idx,
            num_atoms=num_prod_atoms,
            n_std=n_std)

        if len(react_match_tpl_idx) == 0: return
        if len(prod_match_tpl_idx) == 0: return
        if len(react_not_match_tpl_idx) == 0: return
        if len(prod_not_match_tpl_idx) == 0: return

        react_center_sum = react_attn[:, :,
                                      np.array(react_match_tpl_idx)].sum(-1)
        prod_center_sum = prod_attn[:, :, np.array(prod_match_tpl_idx)].sum(-1)

        react_center_mean = react_attn[:, :,
                                       np.array(react_match_tpl_idx)].mean(-1)
        react_center_mean = np.clip(react_center_mean, 1e-6, 1)
        prod_center_mean = prod_attn[:, :,
                                     np.array(prod_match_tpl_idx)].mean(-1)
        prod_center_mean = np.clip(prod_center_mean, 1e-6, 1)

        react_not_center_mean = react_attn[:, :,
                                           np.array(react_not_match_tpl_idx
                                                    )].mean(-1)
        react_not_center_mean = np.clip(react_not_center_mean, 1e-6, 1)
        prod_not_center_mean = prod_attn[:, :,
                                         np.array(prod_not_match_tpl_idx
                                                  )].mean(-1)
        prod_not_center_mean = np.clip(prod_not_center_mean, 1e-6, 1)

        react_center_divide = np.clip(
            react_center_mean / react_not_center_mean, 0.01, 10)
        prod_center_divide = np.clip(prod_center_mean / prod_not_center_mean,
                                     0.01, 10)

        return (react_center_sum, prod_center_sum, react_center_divide,
                prod_center_divide), (react_ol_mat, react_fpr_mat, prod_ol_mat,
                                      prod_fpr_mat, react_cent_acc_two_mat,
                                      react_cent_acc_half_mat,
                                      prod_cent_acc_two_mat,
                                      prod_cent_acc_half_mat)

    return


def get_eval_results(predicted_conditions,
                     attention_weights,
                     input_tokens,
                     ground_truth,
                     rxn_smiles,
                     retro_templates,
                     condition_type_idx,
                     normalize_react_and_prod,
                     n_std,
                     evaluate_mode=False):

    ground_truth_to_eval = list(map(list,
                                    zip(*ground_truth)))[condition_type_idx]
    react_center_sum_collect = np.zeros(attention_weights[0].shape[:2])
    prod_center_sum_collect = np.zeros(attention_weights[0].shape[:2])
    react_center_divide_collect = np.zeros(attention_weights[0].shape[:2])
    prod_center_divide_collect = np.zeros(attention_weights[0].shape[:2])

    react_ol_mat_collect = np.zeros(attention_weights[0].shape[:2])
    react_fpr_mat_collect = np.zeros(attention_weights[0].shape[:2])
    prod_ol_mat_collect = np.zeros(attention_weights[0].shape[:2])
    prod_fpr_mat_collect = np.zeros(attention_weights[0].shape[:2])
    react_cent_acc_two_mat_collect = np.zeros(attention_weights[0].shape[:2])
    react_cent_acc_half_mat_collect = np.zeros(attention_weights[0].shape[:2])
    prod_cent_acc_two_mat_collect = np.zeros(attention_weights[0].shape[:2])
    prod_cent_acc_half_mat_collect = np.zeros(attention_weights[0].shape[:2])

    cnt = 0

    index = [f'layer-{i}' for i in range(attention_weights[0].shape[0])]
    columns = [f'head-{i}' for i in range(attention_weights[0].shape[1])]

    for preds, attn, tokens, gt, rxn, tpl in zip(predicted_conditions,
                                                 attention_weights,
                                                 input_tokens,
                                                 ground_truth_to_eval,
                                                 rxn_smiles, retro_templates):

        results = report_attn_test(preds,
                                   attn,
                                   tokens,
                                   gt,
                                   rxn,
                                   tpl,
                                   condition_type_idx,
                                   normalize_react_and_prod,
                                   n_std=n_std)

        if results:
            (react_center_sum, prod_center_sum, react_center_divide,
             prod_center_divide), (react_ol_mat, react_fpr_mat, prod_ol_mat,
                                   prod_fpr_mat, react_cent_acc_two_mat,
                                   react_cent_acc_half_mat,
                                   prod_cent_acc_two_mat,
                                   prod_cent_acc_half_mat) = results

            react_center_sum_collect += react_center_sum
            prod_center_sum_collect += prod_center_sum
            react_center_divide_collect += react_center_divide
            prod_center_divide_collect += prod_center_divide

            react_ol_mat_collect += react_ol_mat
            react_fpr_mat_collect += react_fpr_mat
            prod_ol_mat_collect += prod_ol_mat
            prod_fpr_mat_collect += prod_fpr_mat
            react_cent_acc_two_mat_collect += react_cent_acc_two_mat
            react_cent_acc_half_mat_collect += react_cent_acc_half_mat
            prod_cent_acc_two_mat_collect += prod_cent_acc_two_mat
            prod_cent_acc_half_mat_collect += prod_cent_acc_half_mat

            cnt += 1

        if (cnt % 1000 == 0) and (cnt != 0) and (not evaluate_mode):

            ############  Center Attention Sum ############
            report_react_center_sum = get_report_df(react_center_sum_collect,
                                                    cnt,
                                                    index=index,
                                                    columns=columns)
            report_prod_center_sum = get_report_df(prod_center_sum_collect,
                                                   cnt,
                                                   index=index,
                                                   columns=columns)

            print('\n############  Center Attention Sum ############')
            print('\nReactants Score Report:')
            print(report_react_center_sum)
            print('\nProducts Score Report:')
            print(report_prod_center_sum)

            ############  Center Attention / Not Center Attention ############
            report_react_center_divide = get_report_df(
                react_center_divide_collect, cnt, index=index, columns=columns)
            report_prod_center_divide = get_report_df(
                prod_center_divide_collect, cnt, index=index, columns=columns)

            print(
                '\n############ Center Attention / Not Center Attention ############'
            )
            print('\nReactants Score Report:')
            print(report_react_center_divide)
            print('\nProducts Score Report:')
            print(report_prod_center_divide)

            ############  Reaction Center Overlap score & False Positive rate & Center Accuracy  ############
            report_react_ol_mat_collect = get_report_df(react_ol_mat_collect,
                                                        cnt,
                                                        index=index,
                                                        columns=columns)
            report_react_fpr_mat_collect = get_report_df(react_fpr_mat_collect,
                                                         cnt,
                                                         index=index,
                                                         columns=columns)
            report_prod_ol_mat_collect = get_report_df(prod_ol_mat_collect,
                                                       cnt,
                                                       index=index,
                                                       columns=columns)
            report_prod_fpr_mat_collect = get_report_df(prod_fpr_mat_collect,
                                                        cnt,
                                                        index=index,
                                                        columns=columns)

            report_react_cent_acc_two_mat_collect = get_report_df(
                react_cent_acc_two_mat_collect,
                cnt,
                index=index,
                columns=columns)
            report_react_cent_acc_half_mat_collect = get_report_df(
                react_cent_acc_half_mat_collect,
                cnt,
                index=index,
                columns=columns)
            report_prod_cent_acc_two_mat_collect = get_report_df(
                prod_cent_acc_two_mat_collect,
                cnt,
                index=index,
                columns=columns)
            report_prod_cent_acc_half_mat_collect = get_report_df(
                prod_cent_acc_half_mat_collect,
                cnt,
                index=index,
                columns=columns)

            print(
                '\n############ Reaction Center Overlap score & False Positive rate & Center Accuracy ############'
            )
            print('\nReactants Center Overlap score Report:')
            print(report_react_ol_mat_collect)
            print('\nReactants Center False Positive rate Report:')
            print(report_react_fpr_mat_collect)
            print('\nReactants Center At Least One Accuracy Report:')
            print(report_react_cent_acc_two_mat_collect)
            print('\nReactants Center Half Accuracy Report:')
            print(report_react_cent_acc_half_mat_collect)

            print('\nProducts Center Overlap score Report:')
            print(report_prod_ol_mat_collect)
            print('\nProducts Center False Positive rate Report:')
            print(report_prod_fpr_mat_collect)
            print('\nProducts Center At Least One Accuracy Report:')
            print(report_prod_cent_acc_two_mat_collect)
            print('\nProducts Center Half Accuracy Report:')
            print(report_prod_cent_acc_half_mat_collect)

    print('\n$$$$$$$$$$$$$$$$$$$$$$$$ end $$$$$$$$$$$$$$$$$$$$$$$$')
    print(f'Count : {cnt}')

    ############  Center Attention Sum ############
    end_react_report_center_sum = get_report_df(react_center_sum_collect,
                                                cnt,
                                                index=index,
                                                columns=columns)
    end_prod_report_center_sum = get_report_df(prod_center_sum_collect,
                                               cnt,
                                               index=index,
                                               columns=columns)

    print('\n############  Center Attention Sum ############')
    print('\nReactants Score Report:')
    print(end_react_report_center_sum)
    print('\nProducts Score Report:')
    print(end_prod_report_center_sum)

    ############ Center Attention / Not Center Attention ############
    end_react_report_center_divide = get_report_df(react_center_divide_collect,
                                                   cnt,
                                                   index=index,
                                                   columns=columns)
    end_prod_report_center_divide = get_report_df(prod_center_divide_collect,
                                                  cnt,
                                                  index=index,
                                                  columns=columns)

    print('############ Center Attention / Not Center Attention ############')
    print('\nReactants Score Report:')
    print(end_react_report_center_divide)
    print('\nProducts Score Report:')
    print(end_prod_report_center_divide)

    ##########  Reaction Center Overlap score & False Positive rate & Center Accuracy ############
    end_report_react_ol_mat_collect = get_report_df(react_ol_mat_collect,
                                                    cnt,
                                                    index=index,
                                                    columns=columns)
    end_report_react_fpr_mat_collect = get_report_df(react_fpr_mat_collect,
                                                     cnt,
                                                     index=index,
                                                     columns=columns)
    end_report_prod_ol_mat_collect = get_report_df(prod_ol_mat_collect,
                                                   cnt,
                                                   index=index,
                                                   columns=columns)
    end_report_prod_fpr_mat_collect = get_report_df(prod_fpr_mat_collect,
                                                    cnt,
                                                    index=index,
                                                    columns=columns)

    end_report_react_cent_acc_two_mat_collect = get_report_df(
        react_cent_acc_two_mat_collect, cnt, index=index, columns=columns)
    end_report_react_cent_acc_half_mat_collect = get_report_df(
        react_cent_acc_half_mat_collect, cnt, index=index, columns=columns)
    end_report_prod_cent_acc_two_mat_collect = get_report_df(
        prod_cent_acc_two_mat_collect, cnt, index=index, columns=columns)
    end_report_prod_cent_acc_half_mat_collect = get_report_df(
        prod_cent_acc_half_mat_collect, cnt, index=index, columns=columns)

    print(
        '\n############  Reaction Center Overlap score & False Positive rate & Center Accuracy ############'
    )
    print('\nReactants Center Overlap score Report:')
    print(end_report_react_ol_mat_collect)
    print('\nReactants Center False Positive rate Report:')
    print(end_report_react_fpr_mat_collect)
    print('\nReactants Center At Least Two Accuracy Report:')
    print(end_report_react_cent_acc_two_mat_collect)
    print('\nReactants Center Half Accuracy Report:')
    print(end_report_react_cent_acc_half_mat_collect)

    print('\nProducts Center Overlap score Report:')
    print(end_report_prod_ol_mat_collect)
    print('\nProducts Center False Positive rate Report:')
    print(end_report_prod_fpr_mat_collect)
    print('\nProducts Center At Least Two Accuracy Report:')
    print(end_report_prod_cent_acc_two_mat_collect)
    print('\nProducts Center Half Accuracy Report:')
    print(end_report_prod_cent_acc_half_mat_collect)

    eval_ol_fpr_sub = end_report_react_ol_mat_collect.values - end_report_react_fpr_mat_collect.values
    ol_fpr_sub_max = eval_ol_fpr_sub.max()
    ol_fpr_sub_max_index = np.argwhere(
        eval_ol_fpr_sub == eval_ol_fpr_sub.max()).tolist()[0]

    best_center_half_accuracy = end_report_react_cent_acc_half_mat_collect.iloc[
        ol_fpr_sub_max_index[0], ol_fpr_sub_max_index[1]]
    best_overlap = end_report_react_ol_mat_collect.iloc[
        ol_fpr_sub_max_index[0], ol_fpr_sub_max_index[1]]
    best_fpr = end_report_react_fpr_mat_collect.iloc[ol_fpr_sub_max_index[0],
                                                     ol_fpr_sub_max_index[1]]

    return best_center_half_accuracy, ol_fpr_sub_max_index, ol_fpr_sub_max, best_overlap, best_fpr


def run_evaluation(evaluation_parameter: dict = {},
                   condition_type_maping: dict = {
                       'c1': 0,
                       's1': 1,
                       's2': 2,
                       'r1': 3,
                       'r2': 4
                   },
                   **kwargs):
    kwargs['evaluate_mode'] = True

    condition_types = evaluation_parameter.get('condition_types',
                                               ['c1', 's1', 's2', 'r1', 'r2'])
    n_std_par = evaluation_parameter.get('n_std', {
        'max': 0.5,
        'min': -0.5,
        'step': 0.1
    })
    n_std_max = n_std_par.get('max', 0.5)
    n_std_min = n_std_par.get('min', -0.5)
    n_std_step = n_std_par.get('step', 0.1)

    n_std_range = np.arange(n_std_min, n_std_max, n_std_step).tolist()

    best_ol_fpr_sub_max = 0.0
    best_center_half_accuracy = 0.0
    best_accuracy_dict = {}

    all_loop_par = []
    for c_type in condition_types:
        for n_std in n_std_range:
            all_loop_par.append((c_type, n_std))

    for par in tqdm(all_loop_par):
        c_type, n_std = par
        condition_type_idx = condition_type_maping[c_type]
        kwargs['condition_type_idx'] = condition_type_idx
        kwargs['n_std'] = n_std
        center_half_accuracy, ol_fpr_sub_max_index, ol_fpr_sub_max, overlap, fpr = get_eval_results(
            **kwargs)

        if ol_fpr_sub_max > best_ol_fpr_sub_max:
            best_ol_fpr_sub_max = ol_fpr_sub_max
            par_dict = {
                'condition_type': c_type,
                'condition_type_idx': condition_type_idx,
                'n_std': n_std,
                'center_half_accuracy': float(center_half_accuracy),
                'max_index': ol_fpr_sub_max_index,
                'overlap_false_positive_rate_sub_max': float(ol_fpr_sub_max),
                'overlap': float(overlap),
                'fpr': float(fpr)
            }
            best_accuracy_dict = par_dict

            print(
                '\n&&&&&&&&&&&&&&&&&&&&&&&&&& Current Best &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
            )
        print(yaml.dump(best_accuracy_dict))

    # for c_type in condition_types:
    #     condition_type_idx = condition_type_maping[c_type]
    #     for n_std in n_std_range:
    #         kwargs['condition_type_idx'] = condition_type_idx
    #         kwargs['n_std'] = n_std
    #         center_half_accuracy, ol_fpr_sub_max_index, ol_fpr_sub_max, overlap, fpr = get_eval_results(**kwargs)

    #         if ol_fpr_sub_max > best_ol_fpr_sub_max:
    #             best_ol_fpr_sub_max = ol_fpr_sub_max
    #             if center_half_accuracy > best_center_half_accuracy:
    #                 best_center_half_accuracy = center_half_accuracy
    #                 par_dict = {
    #                     'condition_type': c_type,
    #                     'condition_type_idx': condition_type_idx,
    #                     'n_std': n_std,
    #                     'center_half_accuracy': float(center_half_accuracy),
    #                     'max_index': ol_fpr_sub_max_index,
    #                     'overlap_false_positive_rate_sub_max' : float(ol_fpr_sub_max),
    #                     'overlap' : float(overlap),
    #                     'fpr': float(fpr)

    #                 }
    #                 best_accuracy_dict = par_dict

    #                 print(
    #                     '\n&&&&&&&&&&&&&&&&&&&&&&&&&& Current Best &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
    #                 )
    #                 print(yaml.dump(best_accuracy_dict))

    return best_accuracy_dict, evaluation_parameter


class ParrotConditionPredictionModelAnalysis(ParrotConditionPredictionModel):

    def __init__(self,
                 model_type,
                 model_name,
                 tokenizer_type=None,
                 tokenizer_name=None,
                 weight=None,
                 args=None,
                 use_cuda=True,
                 cuda_device=-1,
                 freeze_encoder=False,
                 freeze_all_but_one=False,
                 **kwargs):
        super().__init__(model_type, model_name, tokenizer_type,
                         tokenizer_name, weight, args, use_cuda, cuda_device,
                         freeze_encoder, freeze_all_but_one, **kwargs)

    def evaluate_reaction2condition_attn(self,
                                     rxn_smiles: List,
                                     retro_templates: List,
                                     eval_condition_type: str = 'c1',
                                     ground_truth: List = [],
                                     batch_size=8,
                                     normalize_react_and_prod=True,
                                     n_std: float = 0.0,
                                     dataset_flag: str = 'val',
                                     evaluate_mode: bool = False,
                                     eval_config_path: str = None,
                                     eval_results_path: str = None):
        condition_type_maping = {'c1': 0, 's1': 1, 's2': 2, 'r1': 3, 'r2': 4}
        assert eval_condition_type in condition_type_maping
        condition_type_idx = condition_type_maping[eval_condition_type]

        input_df_with_fake_labels = pd.DataFrame({
            'text':
            rxn_smiles,
            'labels': [[0] * 7] * len(rxn_smiles)
        })

        prediction_catch_dir = os.path.join(os.path.dirname(__file__),
                                            '.catch')
        catch_file_path = os.path.join(
            prediction_catch_dir,
            f'cached_prediction_results_{dataset_flag}_cnt_{len(input_df_with_fake_labels)}.pkl'
        )
        os.makedirs(prediction_catch_dir, exist_ok=True)

        if not os.path.exists(catch_file_path):

            predicted_conditions, attention_weights, input_tokens = self.greedy_search_batch_with_attn(
                input_df_with_fake_labels,
                test_batch_size=batch_size,
                normalize=True,
                transpose_end=False)
            cross_attention_weights = attention_weights['cross_attn']
            input_df_with_fake_labels[
                'predicted_conditions'] = predicted_conditions
            input_df_with_fake_labels['attention_weights'] = cross_attention_weights
            input_df_with_fake_labels['input_tokens'] = input_tokens
            input_df_with_fake_labels.to_pickle(catch_file_path)
        else:
            input_df_with_fake_labels = pd.read_pickle(catch_file_path)
            predicted_conditions, cross_attention_weights, input_tokens = input_df_with_fake_labels[
                'predicted_conditions'].tolist(
                ), input_df_with_fake_labels['attention_weights'].tolist(
                ), input_df_with_fake_labels['input_tokens'].tolist()

        if evaluate_mode:
            input_data = {
                'predicted_conditions': predicted_conditions,
                'attention_weights': cross_attention_weights,
                'input_tokens': input_tokens,
                'ground_truth': ground_truth,
                'rxn_smiles': rxn_smiles,
                'retro_templates': retro_templates,
                'condition_type_idx': condition_type_idx,
                'normalize_react_and_prod': normalize_react_and_prod,
                'n_std': n_std
            }

            evaluation_parameter = yaml.load(open(eval_config_path, "r"),
                                             Loader=yaml.FullLoader)

            best_accuracy_dict, evaluation_parameter = run_evaluation(
                evaluation_parameter=evaluation_parameter,
                condition_type_maping=condition_type_maping,
                **input_data)
            print(
                '\n&&&&&&&&&&&&&&&&&&&&&&&&&& Best &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
            )
            print(yaml.dump(best_accuracy_dict))
            print(
                '\n&&&&&&&&&&&&&&&&&&&&&&&&&& Best evaluation parameter &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&'
            )
            print(yaml.dump(evaluation_parameter))

            with open(eval_results_path, 'w', encoding='utf-8') as f:
                yaml.dump(best_accuracy_dict, f)

        else:
            get_eval_results(predicted_conditions,
                             cross_attention_weights,
                             input_tokens,
                             ground_truth,
                             rxn_smiles,
                             retro_templates,
                             condition_type_idx=condition_type_idx,
                             normalize_react_and_prod=normalize_react_and_prod,
                             n_std=n_std)

        return


def run_analysis(parser_args, debug=False):

    print('\n#################################')
    print(yaml.dump(parser_args.__dict__))
    print('\n#################################')

    pandarallel.initialize(nb_workers=parser_args.num_workers,
                           progress_bar=True)
    config = yaml.load(open(parser_args.config_path, "r"),
                       Loader=yaml.FullLoader)

    print(
        '\n########################\Attention Evaluation configs:\n########################\n'
    )
    print(yaml.dump(config))
    print('\n########################\n')
    model_args = config['model_args']
    model_args['pretrained_path'] = os.path.join('..',
                                                 model_args['pretrained_path'])
    model_args['output_dir'] = os.path.join('..', model_args['output_dir'])
    model_args['best_model_dir'] = os.path.join('..',
                                                model_args['best_model_dir'])
    model_args['output_attention'] = model_args.get('output_attention', True)
    model_args['silent'] = True

    dataset_args = config['dataset_args']
    dataset_args['dataset_root'] = os.path.join('..',
                                                dataset_args['dataset_root'])

    try:
        model_args['use_temperature'] = dataset_args['use_temperature']
        print('\nUsing Temperature:', model_args['use_temperature'])
    except:
        print('\nNo temperature information is specified!')

    condition_label_mapping = inference_load(**dataset_args)
    model_args['decoder_args'].update({
        'tgt_vocab_size':
        len(condition_label_mapping[0]),
        'condition_label_mapping':
        condition_label_mapping
    })

    trained_path = model_args['best_model_dir']
    model: ParrotConditionPredictionModelAnalysis = ParrotConditionPredictionModelAnalysis(
        "bert",
        trained_path,
        args=model_args,
        use_cuda=True if parser_args.gpu >= 0 else False,
        cuda_device=parser_args.gpu)

    input_df = pd.read_csv(parser_args.input_path)
    to_canonical = False
    condition_type2cols = {
        'c1': 'catalyst1',
        's1': 'solvent1',
        's2': 'solvent2',
        'r1': 'reagent1',
        'r2': 'reagent2'
    }
    condition_cols = [
        'catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2'
    ]
    for col in condition_cols:
        input_df.loc[input_df[col].isna(), col] = ''

    input_df.loc[input_df['retro_template'].isna(), 'retro_template'] = ''

    if 'dataset' in input_df.columns.tolist():
        input_df = input_df.loc[input_df['dataset'] ==
                                parser_args.dataset_flag]
        # input_df = input_df.loc[input_df[condition_type2cols[
        #     parser_args.eval_condition_type]] != ''].reset_index(drop=True)
        input_df = input_df.loc[input_df['retro_template'] != ''].reset_index(
            drop=True)
    else:
        # input_df = input_df.loc[input_df[condition_type2cols[
        #     parser_args.eval_condition_type]] != ''].reset_index(drop=True)
        input_df = input_df.loc[input_df['retro_template'] != ''].reset_index(
            drop=True)
    if debug:
        input_df = input_df[:100]
    input_rxn_smiles = input_df['canonical_rxn'].tolist()
    retro_templates = input_df['retro_template'].tolist()

    ground_truth_conditions = input_df.parallel_apply(
        lambda row: row[condition_cols].tolist(), axis=1).tolist()

    eval_df = pd.DataFrame({
        'rxn_smiles': input_rxn_smiles,
        'ground_truth_conditions': ground_truth_conditions,
        'retro_template': retro_templates
    })
    if to_canonical:
        print('\nCaonicalize reaction smiles and remove invalid reaction...')
        eval_df['rxn_smiles'] = eval_df['rxn_smiles'].parallel_apply(
            lambda x: caonicalize_rxn_smiles(x))
        eval_df = eval_df.loc[eval_df['rxn_smiles'] != ''].reset_index(
            drop=True)

    config['thread_count'] = parser_args.num_workers

    model.evaluate_reaction2condition_attn(
        eval_df['rxn_smiles'].tolist(),
        retro_templates=eval_df['retro_template'].tolist(),
        eval_condition_type=parser_args.eval_condition_type,
        ground_truth=eval_df['ground_truth_conditions'].tolist(),
        n_std=parser_args.n_std,
        dataset_flag=parser_args.dataset_flag,
        evaluate_mode=parser_args.evaluate_mode,
        eval_config_path=parser_args.eval_config_path,
        eval_results_path=parser_args.eval_results_path)


if __name__ == '__main__':
    parser = ArgumentParser('Test Arguements')
    parser.add_argument('--gpu', default=0, help='GPU device to use', type=int)
    parser.add_argument('--config_path',
                        default='../configs/config_inference_use_uspto.yaml',
                        help='Path to config file',
                        type=str)

    parser.add_argument(
        '--input_path',
        default=
        '../dataset/source_dataset/USPTO_condition_final/USPTO_condition_pred_category_maped_rxn_tpl.csv',
        help='Path to input file (csv)',
        type=str)

    parser.add_argument('--eval_results_path',
                        default='eval_data/eval_results/results_d.yaml',
                        help='Path to input file (csv)',
                        type=str)

    parser.add_argument(
        '--eval_condition_type',
        default='c1',
        help=
        'c1: catalyst,\ns1: solvent1,\ns2: solvent2,\nr1: reagent1,\nr2:reagent2',
        type=str)

    parser.add_argument('--dataset_flag', default='val', type=str)

    parser.add_argument('--evaluate_mode', action='store_true')

    parser.add_argument('--eval_config_path',
                        default='eval_data/par_config/config1.yaml',
                        help='Path to output file (csv)',
                        type=str)
    parser.add_argument('--num_workers',
                        default=10,
                        help='number workers',
                        type=int)
    parser.add_argument('--n_std', default=-0.2, help='n std', type=float)

    parser_args = parser.parse_args()

    debug = True

    run_analysis(parser_args, debug=debug)
