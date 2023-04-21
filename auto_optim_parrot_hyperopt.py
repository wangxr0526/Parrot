import os
import pandas as pd
import torch
import yaml
from hyperopt import fmin, tpe, space_eval, hp
from argparse import ArgumentParser
from models.parrot_model import ParrotConditionPredictionModel
from models.utils import load_dataset

torch.multiprocessing.set_sharing_strategy('file_system')


class SearchByHyperopt:

    def __init__(self, init_config_path, gpu, debug=False) -> None:
        self.init_config = yaml.load(open(init_config_path, "r"),
                                     Loader=yaml.FullLoader)
        model_args = self.init_config['model_args']
        dataset_args = self.init_config['dataset_args']
        try:
            model_args['use_temperature'] = dataset_args['use_temperature']
            print('Using Temperature:', model_args['use_temperature'])
        except:
            print('No temperature information is specified!')

        self.database_df, condition_label_mapping = load_dataset(
            **dataset_args)
        model_args['decoder_args'].update({
            'tgt_vocab_size':
            len(condition_label_mapping[0]),
            'condition_label_mapping':
            condition_label_mapping
        })

        self.train_df = self.database_df.loc[self.database_df['dataset'] ==
                                             'train']
        self.train_df = self.train_df[['canonical_rxn', 'condition_labels'
                                       ]].reset_index(drop=True)
        self.train_df.columns = ['text', 'labels']

        self.eval_df = self.database_df.loc[self.database_df['dataset'] ==
                                            'val'].reset_index(drop=True)
        self.eval_df = self.eval_df[['canonical_rxn', 'condition_labels']]
        self.eval_df.columns = ['text', 'labels']

        if debug:
            self.train_df = self.train_df[:100]
            self.eval_df = self.eval_df[:100]
            model_args['output_dir'] = './out/debug'
            model_args['best_model_dir'] = './outputs/debug'
            model_args['wandb_project'] = 'debug'
            model_args['num_train_epochs'] = 10

        if model_args['pretrained_path']:
            self.pretrained_path = model_args['pretrained_path']
        else:
            self.pretrained_path = None
        self.model_args = model_args

        self.gpu = gpu

    def __call__(self, hyperopt_args):
        batch_size, lr = hyperopt_args
        self.model_args['batch_size'] = batch_size
        self.model_args['lr'] = lr
        self.model_args['output_dir'] = os.path.join(
            self.model_args['output_dir'],
            'batch_size-{}_lr-{}'.format(batch_size, lr))
        self.model_args['best_model_dir'] = os.path.join(
            self.model_args['best_model_dir'],
            'batch_size-{}_lr-{}'.format(batch_size, lr))

        model = ParrotConditionPredictionModel(
            "bert",
            self.pretrained_path,
            args=self.model_args,
            use_cuda=True if self.gpu >= 0 else False,
            cuda_device=self.gpu)

        model.train_model(self.train_df, eval_df=self.eval_df)

        eval_data = pd.read_csv(os.path.join(self.model_args['best_model_dir'],
                                             'eval_results.txt'),
                                sep='=',
                                header=None)
        best_eval_results = eval_data.iloc[0, 1]

        return best_eval_results


def main(parser_args, debug):
    finder = SearchByHyperopt(parser_args.parrot_config_path,
                              parser_args.gpu,
                              debug=debug)

    hyperopt_config = yaml.load(open(parser_args.hyperopt_config_path, "r"),
                                Loader=yaml.FullLoader)

    space = (hp.choice('batch_size', hyperopt_config['batch_size']),
             hp.uniform('lr', float(hyperopt_config['lr']['min']),
                        float(hyperopt_config['lr']['max'])))

    best = fmin(finder, space=space, algo=tpe.suggest, max_evals=20)
    print(best)
    print(space_eval(space, best))


if __name__ == '__main__':
    parser = ArgumentParser('Search Arguements')
    parser.add_argument('--gpu', default=0, help='GPU device to use', type=int)
    parser.add_argument(
        '--parrot_config_path',
        default='configs/config_transfer_to_uspto_suzuki_condition_auto_by_hyperopt.yaml',
        help='Path to config file',
        type=str)

    parser.add_argument(
        '--hyperopt_config_path',
        default='configs/hyperopt/hyperopt_in_transfer_to_uspto_suzuki.yaml',
        help='Path to config file',
        type=str)

    parser_args = parser.parse_args()
    debug = False
    main(parser_args, debug)
