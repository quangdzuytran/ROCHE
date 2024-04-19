import argparse
import numpy as np
import pandas as pd

from multiprocessing.pool import Pool
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from functools import partial

from causa.roche import roche 
from causa.datasets import AN, ANs, LS, LSs, MNU, Tuebingen, SIM, SIMc, SIMG, SIMln, Cha, Multi, Net


BENCHMARKS = {benchmark.__name__: benchmark for benchmark in [AN, ANs, LS, LSs, MNU, Tuebingen, SIM, SIMc, SIMG, SIMln, Cha, Multi, Net]}
METHODS = {
    'ROCHE': partial(roche, independence_test=True, return_function=False, n_steps=5000, verbose=False),
}
TUEBINGEN_IGNORE = [47, 52, 53, 54, 55, 70, 71, 105, 107]
SMOKE_TEST_SIZE = 4


def process(pair_id, data, method, device, verbose):
    dataset = BENCHMARKS[data](pair_id, preprocessor=StandardScaler(), double=True)
    x, y = dataset.cause.flatten().numpy(), dataset.effect.flatten().numpy()
    score = METHODS[method](x, y, device=device)
    if verbose:
        print(pair_id, score, sep='\t')
    return (pair_id, score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str)
    parser.add_argument('--data', type=str, choices=list(BENCHMARKS.keys()))
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    print("-"*40)
    print("Dataset:", args.data)
    print("Method:", args.method)

    n_datasets = BENCHMARKS[args.data].n_datasets
    process_id = partial(process, data=args.data, method=args.method, device=args.device, verbose=args.verbose)
    pair_ids = list(range(1, n_datasets+1))
    if args.data == 'Tuebingen':
        pair_ids = [pair_id for pair_id in pair_ids if pair_id not in TUEBINGEN_IGNORE]
    if args.device == 'cpu' and args.n_jobs > 1:
        with Pool(args.n_jobs) as p:
            res_list = list(p.imap_unordered(process_id, pair_ids))
    else:
        res_list = list(map(process_id, pair_ids))

    results = pd.DataFrame(res_list, columns=['pair_id', 'score']).set_index('pair_id').sort_index()
    results.to_csv(f'{args.result_dir}/{args.method}_{args.data}.csv')

    scores = np.array(results['score'], dtype=np.float64)
    gt = np.ones_like(scores)

    print('Count:', f'{len(scores[scores > 0])}/{len(scores)}')
    print('Acc:', len(scores[scores > 0]) / len(gt))
    print("ROC_AUC:", roc_auc_score(np.concatenate([gt, -gt]), np.concatenate([scores, -scores])))
    
    if args.data == 'Tuebingen':
        tuebingen_meta = pd.read_csv(
            'data/Tuebingen/pairmeta.txt', delim_whitespace=True, 
            header=None, 
            names=['id', 'cause_start', 'cause_end', 'effect_start', 'effect_end', 'weight'],
            index_col=0).astype(float)
        tuebingen_meta.loc[TUEBINGEN_IGNORE, 'weight'] = 0.0
        weights = np.array(tuebingen_meta[tuebingen_meta['weight'] > 0.0]['weight'])[:len(gt)]
        print('Weighted Acc:', np.sum(weights[scores > 0]) / np.sum(weights))
        print("Weighted ROC_AUC:", roc_auc_score(np.concatenate([gt, -gt]), np.concatenate([scores, -scores]), sample_weight=np.concatenate([weights, weights])))