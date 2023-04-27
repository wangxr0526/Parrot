import os
import signal
import sys
import signal
from timeout_decorator import TimeoutError
import pandas as pd
from tqdm.auto import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from analysis_script.calculate_reaction_center_accuracy import get_rxn_center_idx, calculate_score

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))


def timeout(seconds=5):
    def decorator(func):
        def wrapper(*args, **kwargs):
            def signal_handler(signum, frame):
                raise TimeoutError('Function %s timed out after %s seconds' % (func.__name__, seconds))
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result
        return wrapper
    return decorator


def get_center_accuracy(rxn, tpl, pred_tpl):
    if pd.isna(tpl) or pd.isna(pred_tpl): return 0.0, 0.0, 0.0, 0.0
    gt_react_center_idx, _, gt_prod_center_idx, _, num_react_atoms, num_prod_atoms = get_rxn_center_idx(rxn, tpl)
    pred_react_center_idx, _, pred_prod_center_idx, _, _, _ = get_rxn_center_idx(rxn, pred_tpl)

    gt_react_center_idx = set(gt_react_center_idx)
    gt_prod_center_idx = set(gt_prod_center_idx)

    pred_react_center_idx = set(pred_react_center_idx)
    pred_prod_center_idx = set(pred_prod_center_idx)

    rt_overlap_score, rt_false_positive_rate, _, _ = calculate_score(pred_react_center_idx, gt_react_center_idx, num_atoms=num_react_atoms)

    pd_overlap_score, pd_false_positive_rate, _, _ = calculate_score(pred_prod_center_idx, gt_prod_center_idx, num_atoms=num_react_atoms)

    return rt_overlap_score, rt_false_positive_rate, pd_overlap_score, pd_false_positive_rate

def get_center_accuracy_in_df(df):

    all_rt_overlap_score = 0.0
    all_rt_false_positive_rate = 0.0
    all_pd_overlap_score = 0.0
    all_pd_false_positive_rate = 0.0
    cnt = 0
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        rxn, tpl, pred_tpl = row[['canonical_rxn', 'retro_template', 'self_encoder_mapped_tpl']].tolist()

        rt_overlap_score, rt_false_positive_rate, pd_overlap_score, pd_false_positive_rate = get_center_accuracy(rxn, tpl, pred_tpl)

        all_rt_overlap_score += rt_overlap_score
        all_rt_false_positive_rate += rt_false_positive_rate
        all_pd_overlap_score += pd_overlap_score
        all_pd_false_positive_rate += pd_false_positive_rate
        cnt += 1
    
    results = {
        'reactants center overlap score': all_rt_overlap_score/cnt,
        'reactants center fpr score': all_rt_false_positive_rate/cnt,
        'products center overlap score': all_pd_overlap_score/cnt,
        'products center fpr score': all_pd_false_positive_rate/cnt,
    }

    return results



    


def get_results(result):
    if pd.isna(result):
        return 
    return result['mapped_rxn']


if __name__ == '__main__':

    debug = False

    input_path = '../dataset/source_dataset/USPTO_condition_final/USPTO_condition_pred_category_maped_rxn_tpl.csv'
    save_path = input_path.replace('.csv', '_self_mapped.csv')

    if not os.path.exists(save_path):
        from analysis_script.evaluate_mapper_with_reaction_self_attention import Mapper

        mapper = Mapper(config_path='../configs/config_inference_use_uspto.yaml')



        input_df = pd.read_csv(input_path)

        eval_df = input_df.loc[input_df['dataset'] == 'val']
        print('eval dataframe', eval_df.shape)
        test_df = input_df.loc[input_df['dataset'] == 'test']
        print('test dataframe', test_df.shape)

        if debug:
            eval_df = eval_df.sample(n=100)
            test_df = test_df.sample(n=100)

        input_df.loc[eval_df.index, 'mapper_results'] = mapper.get_batched_attention_guided_atom_maps(
            eval_df['canonical_rxn'].tolist(),
            debug=debug,
            batch_size=128)
        # input_df.loc[eval_df.index, '_canonical_rxn'] = eval_df['canonical_rxn'].tolist()




        input_df.loc[test_df.index, 'mapper_results'] = mapper.get_batched_attention_guided_atom_maps(
            test_df['canonical_rxn'].tolist(),
            debug=debug,
            batch_size=128)
        # input_df.loc[test_df.index, '_canonical_rxn'] = test_df['canonical_rxn'].tolist()


        input_df['self_encoder_mapped_rxn'] = input_df['mapper_results'].apply(lambda x:get_results(x))
        input_df = input_df.drop(['mapper_results'], axis=1)
        if not debug:
            input_df.to_csv(save_path, index=False)
    
    else:
        input_df = pd.read_csv(save_path)
    
    if 'self_encoder_mapped_tpl' not in input_df.columns.tolist():
        from pandarallel import pandarallel
        pandarallel.initialize(nb_workers=12, progress_bar=True)
        from preprocess_script.uspto_script.extract_std_template import get_templates

        @timeout(5)
        def get_templates_timeout(rxn_smi, 
                  prec,     
                  add_brackets=True):
            results =  get_templates(rxn_smi, prec, add_brackets)
            return results

        def get_templates_with_timeout(rxn_smi, 
                  prec,     
                  add_brackets=True):
            try:
                return get_templates_timeout(rxn_smi, prec, add_brackets)
            except:
                return ''


        eval_df = input_df.loc[input_df['dataset'] == 'val']

        test_df = input_df.loc[input_df['dataset'] == 'test']

        if debug:
            eval_df = eval_df.sample(n=100)
            test_df = test_df.sample(n=100)

        input_df.loc[eval_df.index, 'self_encoder_mapped_tpl'] = eval_df.parallel_apply(lambda row: get_templates_with_timeout(row['self_encoder_mapped_rxn'], row['precursors'], add_brackets=False), axis=1) 
        
        
        input_df.loc[test_df.index, 'self_encoder_mapped_tpl'] = test_df.parallel_apply(lambda row: get_templates_with_timeout(row['self_encoder_mapped_rxn'], row['precursors'], add_brackets=False), axis=1)

        if not debug:
            input_df.to_csv(save_path, index=False)

    else:
        input_df = pd.read_csv(save_path)
        eval_df = input_df.loc[input_df['dataset'] == 'val']
        print('eval template accuracy:')
        print(((eval_df['self_encoder_mapped_tpl'] == eval_df['retro_template']) & (~eval_df['self_encoder_mapped_tpl'].isna())).sum()/len(eval_df))
        print(get_center_accuracy_in_df(eval_df))

        test_df = input_df.loc[input_df['dataset'] == 'test']
        print('test template accuracy:')
        print(((test_df['self_encoder_mapped_tpl'] == test_df['retro_template']) & (~test_df['self_encoder_mapped_tpl'].isna())).sum()/len(test_df))
        print(get_center_accuracy_in_df(test_df))
        









        exit(0)



    


    pass