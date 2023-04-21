import pickle
import os
from collections import OrderedDict
from pickletools import optimize
import random
import time
from matplotlib import use
from matplotlib.pyplot import savefig
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm, trange
from baseline_condition_model import RXNConditionModel, create_rxn_Morgan2FP_separately
import wandb
import yaml
from torch.utils.data import Dataset, DataLoader
import json
from train_baseline_model import ConditionDataset, caculate_accuracy, get_one_hot_input, load_model, load_dataset

# from torch.utils.tensorboard import SummaryWriter

# wandb.init(project="baseline_condition_model", entity="wangxr")

work_space = os.path.dirname(__file__)


def load_test_dataset(data_root, database_name, label_dict, use_temperature=False):
         
    fp_size = 16384
    print('Loading condition data from {}'.format(data_root))

    print('Loading test database dataframe...')
    database_df = pd.read_csv(os.path.join(data_root, f'{database_name}.csv'))
    print('Loaded {} data'.format(len(database_df)))

    print('caculating fps...')
    canonical_rxn = database_df.canonical_rxn.tolist()
    prod_fps = []
    rxn_fps = []
    for rxn in tqdm(canonical_rxn):
        rsmi, psmi = rxn.split('>>')
        [pfp, rfp] = create_rxn_Morgan2FP_separately(
            rsmi, psmi, rxnfpsize=fp_size, pfpsize=fp_size, useFeatures=False, calculate_rfp=True, useChirality=True)
        rxn_fp = pfp - rfp
        prod_fps.append(pfp)
        rxn_fps.append(rxn_fp)
    prod_fps = np.array(prod_fps)
    rxn_fps = np.array(rxn_fps)
    print('prod_fps shape:', prod_fps.shape)
    print('rxn_fps shape:', rxn_fps.shape)
    print('########################################################')
    prod_fps_dict = {}
    rxn_fps_dict = {}

    # for dataset in list(set(database_df['dataset'].tolist())):
    dataset_prod_fps = prod_fps
    prod_fps_dict['test'] = dataset_prod_fps
    dataset_rxn_fps = rxn_fps
    rxn_fps_dict['test'] = dataset_rxn_fps
    print('{} prod_fps shape: {}'.format('test', dataset_prod_fps.shape))
    print('{} rxn_fps shape: {}'.format('test', dataset_rxn_fps.shape))

    new_label_dict = OrderedDict()
    for name in ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']:
        [_label_dic, name_labels] = label_dict[name]
        print('Condition name: {}, categories: {}'.format(
            name, len(_label_dic[0])))
        name_data = database_df[name]
        name_data[pd.isna(name_data)] = ''
        name_labels = {}
        # for dataset in list(set(database_df['dataset'].tolist())):
        condition2label = _label_dic[1]
        name_labels['test'] = [condition2label[x]
                                for x in name_data.tolist()]
        name_labels['test'] = np.array(name_labels['test'])
        print('{}: {}'.format('test', len(name_labels['test'])))

        print('########################################################')
        new_label_dict[name] = [_label_dic, name_labels]
    fps_dict = {'prod_fps': prod_fps_dict, 'rxn_fps': rxn_fps_dict}
    return fps_dict, new_label_dict

work_space = os.path.dirname(__file__)


def load_dataset_label(data_root, database_name, use_temperature=False):
    print('Loading condition data from {}'.format(data_root))

    print('Loading database dataframe...')


    # print('Loading caculated fps...')
    # prod_fps = np.load(os.path.join(
    #     data_root, f'{database_name}_prod_fps.npz'))['fps']
    # rxn_fps = np.load(os.path.join(
    #     data_root, f'{database_name}_rxn_fps.npz'))['fps']
    # print('prod_fps shape:', prod_fps.shape)
    # print('rxn_fps shape:', rxn_fps.shape)

    if use_temperature:
        temperature_dict = {}

    print('Loading label dict...')
    label_dict = OrderedDict()
    for name in ['catalyst1', 'solvent1', 'solvent2', 'reagent1', 'reagent2']:
        with open(os.path.join(data_root, '{}_{}.pkl'.format(database_name, name)), 'rb') as f:
            _label_dic = pickle.load(f)
        print('Condition name: {}, categories: {}'.format(
            name, len(_label_dic[0])))

        label_dict[name] = [_label_dic, None]
        print('########################################################')

    fps_dict = None
    return fps_dict, label_dict

def caculate_accuracy_supercls(model, data_loader, device, condition2label, topk_rank_thres=None, save_topk_path=None, use_temperature=False, top_fname=None, super_class_dicts=None, topk_get=[1, 3, 5, 10, 15],condition_to_calculate=['c1', 's1', 's2', 'r1', 'r2']):
    print('Caculating validataion topk accuracy...')
    if not topk_rank_thres:
        topk_rank_thres = {
            'c1': 1,
            's1': 3,
            's2': 1,
            'r1': 5,
            'r2': 1,
        }

    def get_accuracy_for_one(one_pred, one_ground_truth, topk_get=[1, 3, 5, 10, 15], condition_to_calculate=['c1', 's1', 's2', 'r1', 'r2']):
        
        condition_item2cols = {
            'c1':0, 's1':1, 's2':2, 'r1':3, 'r2':4
        }
        
        calculate_cols = [condition_item2cols[x] for x in condition_to_calculate]
        
        repeat_number = one_pred.size(0)
        label2condition = {}
        for key in condition2label:
            label2condition[key] = {v:k for k,v in condition2label[key].items()}
        hit_mat = one_ground_truth.unsqueeze(
            0).repeat(repeat_number, 1) == one_pred
        hit_mat = hit_mat[:, calculate_cols]
        overall_hit_mat = hit_mat.sum(1) == hit_mat.size(1)
        topk_hit_df = pd.DataFrame()
        for k in topk_get:
            hit_mat_k = hit_mat[:k, :]
            overall_hit_mat_k = overall_hit_mat[:k]
            topk_hit = []
            for col_idx in range(hit_mat.size(1)):
                if hit_mat_k[:, col_idx].sum() != 0:
                    topk_hit.append(1)
                else:
                    topk_hit.append(0)
            if overall_hit_mat_k.sum() != 0:
                topk_hit.append(1)
            else:
                topk_hit.append(0)
            topk_hit_df[k] = topk_hit
        # topk_hit_df.index = ['c1', 's1', 's2', 'r1', 'r2']
        return topk_hit_df

    model.eval()
    model.not_softmax_out = False  # 输出通过softmax激活函数
    topk_acc_mat = np.zeros((len(condition_to_calculate) + 1, 5))
    closest_erro = 0.0
    for data in tqdm(data_loader):

        with torch.no_grad():

            data = [x.to(device) for x in data]
            if not use_temperature:
                pfp, rfp, c1, s1, s2, r1, r2 = data
            else:
                pfp, rfp, c1, s1, s2, r1, r2, t = data

            fp_emb = model.fp_func(pfp, rfp)
            one_batch_ground_truth = torch.cat([c1.unsqueeze(1), s1.unsqueeze(1), s2.unsqueeze(
                1), r1.unsqueeze(1), r2.unsqueeze(1)], dim=-1)
            # if device != torch.device('cpu'):
            #     one_batch_ground_truth = torch.cat([c1.unsqueeze(1), s1.unsqueeze(1), s2.unsqueeze(
            #         1), r1.unsqueeze(1), r2.unsqueeze(1)], dim=-1).cpu().numpy()
            # else:
            #     one_batch_ground_truth = torch.cat([c1.unsqueeze(1), s1.unsqueeze(1), s2.unsqueeze(
            #         1), r1.unsqueeze(1), r2.unsqueeze(1)], dim=-1).numpy()

            # one_batch_ground_truth_df = pd.DataFrame(one_batch_ground_truth)
            # one_batch_ground_truth_df.columns = [
            #     'ground_truth-c1', 'ground_truth-s1', 'ground_truth-s2', 'ground_truth-r1', 'ground_truth-r2', ]

            # teached_c1_pred = model.c1_func(fp_emb)
            # _, teached_c1_top_15 = teached_c1_pred.squeeze().topk(15)
            # input_c1 = get_one_hot_input(
            #     c1, dim=len(condition2label['catalyst1']))
            # teached_s1_pred = model.s1_func(fp_emb, input_c1)
            # _, teached_s1_top_15 = teached_s1_pred.squeeze().topk(15)
            # input_s1 = get_one_hot_input(
            #     s1, dim=len(condition2label['solvent1']))
            # teached_s2_pred = model.s2_func(fp_emb, input_c1, input_s1)
            # _, teached_s2_top_15 = teached_s2_pred.squeeze().topk(15)
            # input_s2 = get_one_hot_input(
            #     s2, dim=len(condition2label['solvent2']))
            # teached_r1_pred = model.r1_func(
            #     fp_emb, input_c1, input_s1, input_s2)
            # _, teached_r1_top_15 = teached_r1_pred.squeeze().topk(15)
            # input_r1 = get_one_hot_input(
            #     r1, dim=len(condition2label['reagent1']))
            # teached_r2_pred = model.r2_func(fp_emb, input_c1,
            #                                 input_s1, input_s2, input_r1)
            # _, teached_r2_top_15 = teached_r2_pred.squeeze().topk(15)

            # if device != torch.device('cpu'):
            #     teached_top_15 = torch.cat([teached_c1_top_15, teached_s1_top_15, teached_s2_top_15,
            #                                teached_r1_top_15, teached_r2_top_15, ], dim=-1).cpu().numpy()
            # else:
            #     teached_top_15 = torch.cat(
            #         [teached_c1_top_15, teached_s1_top_15, teached_s2_top_15, teached_r1_top_15, teached_r2_top_15, ], dim=-1).numpy()
            # one_batch_prediction_df = pd.DataFrame(teached_top_15)
            # one_batch_prediction_df.columns = ['teached_c1_top_{}'.format(x+1) for x in range(15)] + \
            #     ['teached_s1_top_{}'.format(x+1) for x in range(15)] + ['teached_s2_top_{}'.format(x+1) for x in range(15)] + \
            #     ['teached_r1_top_{}'.format(
            #         x+1) for x in range(15)] + ['teached_r2_top_{}'.format(x+1) for x in range(15)]
            # # one_batch_prediction_df = pd.concat([one_batch_prediction_df, one_batch_prediction_df_teached_top_15], axis=1)

            one_batch_preds = []
            one_batch_scores = []
            one_batch_t_preds = []

            c1_preds = model.c1_func(fp_emb)
            c1_scores, c1_cdts = c1_preds.squeeze().topk(topk_rank_thres['c1'])
            for c1_top in range(c1_cdts.size(-1)):
                # if device != torch.device('cpu'):
                #     pred_list = c1_cdts[:, c1_top].cpu().numpy().tolist()
                # else:
                #     pred_list = c1_cdts[:, c1_top].numpy().tolist()

                # one_batch_prediction_df['c1_top-{}'.format(c1_top+1)
                #                         ] = pred_list
                # c1_pred_list = pred_list
                c1_pred = c1_cdts[:, c1_top]
                c1_score = c1_scores[:, c1_top]

                c1_input = get_one_hot_input(
                    c1_pred, len(condition2label['catalyst1']))

                s1_preds = model.s1_func(fp_emb, c1_input)
                s1_scores, s1_cdts = s1_preds.squeeze().topk(
                    topk_rank_thres['s1'])
                for s1_top in range(s1_cdts.size(-1)):
                    # if device != torch.device('cpu'):
                    #     pred_list = s1_cdts[:, s1_top].cpu().numpy().tolist()
                    # else:
                    #     pred_list = s1_cdts[:, s1_top].numpy().tolist()
                    # one_batch_prediction_df['c1_top-{}_s1_top-{}'.format(c1_top+1, s1_top+1)
                    #                         ] = pred_list
                    # s1_pred_list = pred_list
                    s1_pred = s1_cdts[:, s1_top]
                    s1_score = s1_scores[:, s1_top]
                    s1_input = get_one_hot_input(
                        s1_pred, len(condition2label['solvent1']))

                    s2_preds = model.s2_func(fp_emb, c1_input, s1_input)
                    s2_scores, s2_cdts = s2_preds.squeeze().topk(
                        topk_rank_thres['s2'])
                    for s2_top in range(s2_cdts.size(-1)):
                        # if device != torch.device('cpu'):
                        #     pred_list = s2_cdts[:,
                        #                         s2_top].cpu().numpy().tolist()
                        # else:
                        #     pred_list = s2_cdts[:, s2_top].numpy().tolist()
                        # one_batch_prediction_df['c1_top-{}_s1_top-{}_s2_top-{}'.format(c1_top+1, s1_top+1, s2_top+1)
                        #                         ] = pred_list
                        # s2_pred_list = pred_list
                        s2_pred = s2_cdts[:, s2_top]
                        s2_score = s2_scores[:, s2_top]
                        s2_input = get_one_hot_input(
                            s2_pred, len(condition2label['solvent2']))

                        r1_preds = model.r1_func(
                            fp_emb, c1_input, s1_input, s2_input)
                        r1_scores, r1_cdts = r1_preds.squeeze().topk(
                            topk_rank_thres['r1'])
                        for r1_top in range(r1_cdts.size(-1)):
                            # if device != torch.device('cpu'):
                            #     pred_list = r1_cdts[:, r1_top].cpu(
                            #     ).numpy().tolist()
                            # else:
                            #     pred_list = r1_cdts[:, r1_top].numpy().tolist()
                            # one_batch_prediction_df['c1_top-{}_s1_top-{}_s2_top-{}_r1_top-{}'.format(c1_top+1, s1_top+1, s2_top+1, r1_top+1)
                            #                         ] = pred_list
                            # r1_pred_list = pred_list
                            r1_pred = r1_cdts[:, r1_top]
                            r1_score = r1_scores[:, r1_top]
                            r1_input = get_one_hot_input(
                                r1_pred, len(condition2label['reagent1']))

                            r2_preds = model.r2_func(
                                fp_emb, c1_input, s1_input, s2_input, r1_input)
                            r2_scores, r2_cdts = r2_preds.squeeze().topk(
                                topk_rank_thres['r2'])
                            for r2_top in range(r2_cdts.size(-1)):
                                # if device != torch.device('cpu'):
                                #     pred_list = r2_cdts[:, r2_top].cpu(
                                #     ).numpy().tolist()
                                # else:
                                #     pred_list = r2_cdts[:,
                                #                         r2_top].numpy().tolist()
                                r2_pred = r2_cdts[:, r2_top]
                                r2_score = r2_scores[:, r2_top]
                                # one_batch_prediction_df['c1_top-{}_s1_top-{}_s2_top-{}_r1_top-{}_r2_top-{}'.format(c1_top+1, s1_top+1, s2_top+1, r1_top+1, r2_top+1)
                                #                         ] = pred_list
                                # r2_pred_list = pred_list

                                one_pred = torch.cat([c1_pred.unsqueeze(1), s1_pred.unsqueeze(
                                    1), s2_pred.unsqueeze(1), r1_pred.unsqueeze(1), r2_pred.unsqueeze(1)], dim=-1)
                                one_score = c1_score * s1_score * s2_score * r1_score * r2_score
                                one_batch_preds.append(one_pred)
                                one_batch_scores.append(one_score)

                                if use_temperature:
                                    r2_input = get_one_hot_input(
                                        r2_pred, len(condition2label['reagent2']))
                                    t_preds = model.T_func(
                                        fp_emb, c1_input, s1_input, s2_input, r1_input, r2_input)
                                    t_preds = t_preds.squeeze()
                                    one_batch_t_preds.append(t_preds)

            one_batch_preds = torch.cat(
                [x.unsqueeze(0) for x in one_batch_preds], dim=0)
            one_batch_scores = torch.cat(
                [x.unsqueeze(0) for x in one_batch_scores], dim=0)
            if use_temperature:
                one_batch_t_preds = torch.cat(
                    [x.unsqueeze(0) for x in one_batch_t_preds], dim=0
                )

            sorted_one_batch_preds = []
            sorted_one_batch_t_preds = []
            batch_number = pfp.size(0)
            for n in range(batch_number):
                sorted_one_batch_preds.append(
                    one_batch_preds[one_batch_scores[:, n].argsort(
                        descending=True), n, :]
                )
                if use_temperature:
                    sorted_one_batch_t_preds.append(
                        one_batch_t_preds[one_batch_scores[:,
                                                           n].argsort(descending=True), n]
                    )
            sorted_one_batch_preds = torch.cat(
                [x.unsqueeze(0) for x in sorted_one_batch_preds], dim=0)
            if use_temperature:
                sorted_one_batch_t_preds = torch.cat(
                    [x.unsqueeze(0) for x in sorted_one_batch_t_preds], dim=0)
                closest_one_batch_t_preds = sorted_one_batch_t_preds[:, 0]
                closest_erro += torch.abs(closest_one_batch_t_preds - t).sum()


            one_batch_topk_acc_mat = np.zeros(topk_acc_mat.shape)
            # topk_get = [1, 3, 5, 10, 15]
            for idx in range(sorted_one_batch_preds.size(0)):
                one_sample_results = sorted_one_batch_preds[idx]
                one_sample_results_supercls = []
                for pred in one_sample_results.tolist():
                    pred_tokens = [
                        label2condition['catalyst1'][pred[0]],
                        label2condition['solvent1'][pred[1]],
                        label2condition['solvent2'][pred[2]],
                        label2condition['reagent1'][pred[3]],
                        label2condition['reagent2'][pred[4]],
                        ]
                    pred_supercls = [
                        condition2label['catalyst1'][pred_tokens[0]],
                        super_class_dicts['solvent'][pred_tokens[1]] + 500,
                        super_class_dicts['solvent'][pred_tokens[2]] + 1000,
                        super_class_dicts['reagent'][pred_tokens[3]] + 1500,
                        super_class_dicts['reagent'][pred_tokens[4]] + 2000,
                        
                    ]
                    one_sample_results_supercls.append(pred_supercls)
                
                one_label = one_batch_ground_truth[idx].tolist()
                one_tokens = [                        
                    label2condition['catalyst1'][one_label[0]],
                    label2condition['solvent1'][one_label[1]],
                    label2condition['solvent2'][one_label[2]],
                    label2condition['reagent1'][one_label[3]],
                    label2condition['reagent2'][one_label[4]],
                    ]
                one_label_supercls = [
                    condition2label['catalyst1'][one_tokens[0]],
                    super_class_dicts['solvent'][one_tokens[1]] + 500,
                    super_class_dicts['solvent'][one_tokens[2]] + 1000,
                    super_class_dicts['reagent'][one_tokens[3]] + 1500,
                    super_class_dicts['reagent'][one_tokens[4]] + 2000,
                ]
                labels_supercls = one_label_supercls
                
                one_sample_results_supercls = torch.tensor(one_sample_results_supercls, device=sorted_one_batch_preds.device)
                labels_supercls = torch.tensor(labels_supercls, device=sorted_one_batch_preds.device)
                topk_hit_df = get_accuracy_for_one(
                    one_sample_results_supercls, labels_supercls, topk_get=topk_get, condition_to_calculate=condition_to_calculate)
                one_batch_topk_acc_mat += topk_hit_df.values
            topk_acc_mat += one_batch_topk_acc_mat
    topk_acc_mat /= len(data_loader.dataset)
    topk_acc_df = pd.DataFrame(topk_acc_mat)

    topk_acc_df.columns = [f'top-{k} accuracy' for k in topk_get]
    # topk_acc_df.index = ['c1', 's1', 's2', 'r1', 'r2', 'overall']
    topk_acc_df.index = condition_to_calculate + ['overall']
    if use_temperature:
        closest_temp_mae = (closest_erro / len(data_loader.dataset)).item()
        topk_acc_df.loc['closest_pred_temp_mae'] = [
            closest_temp_mae] * len(topk_acc_df.columns)
    topk_acc_df = topk_acc_df.round(4)
    print(topk_acc_df)
    if top_fname:
        
        if save_topk_path:
            topk_acc_df.to_csv(os.path.join(save_topk_path, top_fname))
    else:
        if save_topk_path:
            topk_acc_df.to_csv(os.path.join(save_topk_path, 'test_accuacy.csv'))
    return topk_acc_df


if __name__ == '__main__':
    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    config = yaml.load(open('./beseline_config/baseline_config_test_for_ood_supercls_catalyst_na.yaml', "r"),
                       Loader=yaml.FullLoader)

    device = torch.device('cuda:{}'.format(
        config['gpu'])) if config['gpu'] >= 0 else torch.device('cpu')
    time_str = time.strftime('%Y-%m-%d_%Hh-%Mm-%Ss',
                             time.localtime(time.time()))
    # writer = SummaryWriter(f'runs/{time_str}')

    final_condition_data_path = config['database_path']
    test_final_condition_data_path = config['test_database_path']
    _, label_dict = load_dataset_label(
        final_condition_data_path, config['database_name'], use_temperature=config['use_temperature'])
    test_fps_dict, label_dict = load_test_dataset(test_final_condition_data_path, config['test_database_name'], label_dict = label_dict, use_temperature=False)

    test_dataset = ConditionDataset(
        fps_dict=test_fps_dict, label_dict=label_dict, dataset='test', use_temperature=config['use_temperature'])

    condition2label = test_dataset.condition2label
    label2condition = test_dataset.label2condition
    fp_dim = test_dataset.fp_dim
    c1_dim = test_dataset.c1_dim
    s1_dim = test_dataset.s1_dim
    s2_dim = test_dataset.s2_dim
    r1_dim = test_dataset.r1_dim
    r2_dim = test_dataset.r2_dim

    config['fp_dim'] = fp_dim
    config['c1_dim'] = c1_dim
    config['s1_dim'] = s1_dim
    config['s2_dim'] = s2_dim
    config['r1_dim'] = r1_dim
    config['r2_dim'] = r2_dim


    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'], shuffle=False)

    model = RXNConditionModel(
        fp_dim=fp_dim,
        h_dim=config['h_dim'],
        dropout_rate=config['dropout_rate'],
        c1_dim=c1_dim,
        s1_dim=s1_dim,
        s2_dim=s2_dim,
        r1_dim=r1_dim,
        r2_dim=r2_dim,
        # is_train=True,
    ).to(device)
    loss_fn_list = [
        torch.nn.CrossEntropyLoss(),
        torch.nn.MSELoss()
    ]
    # loss_fn = torch.nn.CrossEntropyLoss()
    print(config)
    print(model)

    print('Testing model...')
    model = load_model(model, os.path.join(
        config['model_path'], config['model_name']))
    model = model.to(device)
    if config['using_super_class_to_test']:
        print('using super classification to test')
        with open('../data/uspto_script/condition_classfication_data/uspto_reagent_to_cls_idx.json', 'r', encoding='utf-8') as f:
            uspto_reagent_to_cls_idx = json.load(f)
        with open('../data/uspto_script/condition_classfication_data/uspto_solvent_to_cls_idx.json', 'r', encoding='utf-8') as f:
            uspto_solvent_to_cls_idx = json.load(f)
        super_class_dicts = {
            'solvent': uspto_solvent_to_cls_idx,
            'reagent': uspto_reagent_to_cls_idx,
        }
        
        caculate_accuracy_supercls(model,
                            test_loader,
                            device=device,
                            condition2label=condition2label,
                            super_class_dicts=super_class_dicts,
                            topk_rank_thres=config['topk_rank_thres'] if 'topk_rank_thres' in config else {
                                'c1': 2,
                                's1': 3,
                                's2': 1,
                                'r1': 3,
                                'r2': 1,
                            },
                            save_topk_path=os.path.join(config['model_path'], config['model_name']), use_temperature=config['use_temperature'], top_fname=config['top_fname'],
                            condition_to_calculate=config['condition_to_calculate'] if 'condition_to_calculate' in config else ['c1', 's1', 's2', 'r1', 'r2'])
    
    else:
        caculate_accuracy(model,
                            test_loader,
                            device=device,
                            condition2label=condition2label,
                            topk_rank_thres=config['topk_rank_thres'] if 'topk_rank_thres' in config else {
                                'c1': 2,
                                's1': 3,
                                's2': 1,
                                'r1': 3,
                                'r2': 1,
                            },
                            save_topk_path=os.path.join(config['model_path'], config['model_name']), use_temperature=config['use_temperature'], top_fname=config['top_fname'],
                            condition_to_calculate=config['condition_to_calculate'] if 'condition_to_calculate' in config else ['c1', 's1', 's2', 'r1', 'r2'])

        # writer.close()
