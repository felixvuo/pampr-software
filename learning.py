import argparse
import itertools as it
import numpy as np
import pandas as pd
import pandas.api.types as pdt
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestRegressor as Reg
# from sklearn.ensemble import ExtraTreesRegressor as Reg
from sklearn.metrics import mean_absolute_percentage_error as maperr
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

from dataset import NON_FEAT_COLS
import util as ut

PREDICT_COLS = ['score', 'plan_exec_time', 'time_plus_tail']
DONT_TRAIN_ON = ['fr_idx', 'to_idx', 'round_id'] + PREDICT_COLS


def main():
    ap = argparse.ArgumentParser(
        description="train (and evaluate) model"
    )
    ap.add_argument('path_to_conf', type=Path, default=None, help="where's the trial config yaml file?")
    ap.add_argument('--load-saved-models', '-l', action='store_true', default=False,
                    help="Use saved trained models (for eval)")
    args = ap.parse_args()
    conf, _, _, _ = ut.init_configuration(args.path_to_conf)
    trial_dir = args.path_to_conf.parent

    print("Loading stats and features")
    df_feats = pd.read_csv(args.path_to_conf.parent / "feats.csv.gz").convert_dtypes()
    df_stats = pd.read_csv(args.path_to_conf.parent / "stats.csv.gz").convert_dtypes()
    print("Labelling data...")
    df_combi = label_data(df_feats, df_stats, conf)
    print(f"Combined dataset:\n{df_combi}")
    eval_frames = []
    pred_frames = []
    all_single_bests = []
    for seed in range(conf['n_train_test_splits']):
        print(f"~~~~~ SEED = {seed:>4d} ~~~~~")
        test_frac = conf.get('test_frac_endpoints',False)
        opts = dict(test_frac=test_frac) if test_frac else {}
        df_train, df_test = train_test_split(df_combi, seed=seed, **opts)
        _fname = f'training-{seed:04d}.pickle'
        if args.load_saved_models:
            with open(_fname, 'rb') as f:
                models, scaler = pickle.load(f)
        else:
            models, scaler = train_models(df_train, seed=seed)
            with open(_fname, 'wb') as f:
                pickle.dump((models, scaler), f)
        preds = make_predictions(df_test, models, scaler)

        # first save the raw predictions
        _cols = ['fr_idx', 'to_idx', 'round_id', 'ptype_id', 'conf_id']
        df_preds = df_test.loc[:,_cols]
        df_preds['seed'] = seed
        for c in PREDICT_COLS:
            df_preds[c] = preds[c]

        # predictions are same for each round - they came from the same features
        pred_frames.append(df_preds.drop(columns=['round_id']).drop_duplicates())

        # summarise and evaluate
        single_bests = decide_single_bests(df_train)
        for measure in PREDICT_COLS:
            all_single_bests.append(dict(
                split = seed,
                measure = measure,
                ptype_id = single_bests[measure][0],
                conf_id = single_bests[measure][1]
            ))
        df_eval = evaluate_preds(df_test, preds, single_bests)
        df_eval['seed'] = seed
        eval_frames.append(df_eval)
    all_eval = pd.concat(eval_frames, ignore_index=True)
    print(f"Aggregated stats:\n{all_eval.describe().T}")
    all_eval.to_csv(trial_dir / 'regression-eval.csv.gz', index=False)
    all_eval.describe().T.to_csv(trial_dir / 'regression-summary.csv', index=True)
    pd.DataFrame(all_single_bests).to_csv(trial_dir / 'single-bests.csv', index = None)
    pd.concat(pred_frames, ignore_index=True).to_csv(trial_dir / 'preds.csv.gz', index=None)
    


def label_data(df_feats, df_stats, conf):
    """Put together the experimental results `labels` with the features

    Two label columns are created:
    
    - `score` records the time score which incorporates variability measures

    - `plan_exec_time` records the combined planning and execution time for each trial

    - `time_plus_tail` the plan + exec time + (95th - 50th percentile)

    """
    # implement penalty for timed-out plans and fill in blank for plan_exec_time
    timeout = conf['planning_timeout_sec']
    penalty = conf['planning_timeout_penalty']
    to_penalise = (df_stats.time_to_plan > timeout) | (df_stats.path_n_points.isna())
    df_stats.loc[to_penalise, 'time_to_plan'] = timeout * penalty
    df_stats.loc[to_penalise, 'path_exec_time'] = df_stats.path_exec_time.max()

    index_cols = ['ptype_id', 'conf_id', 'query_sta', 'query_end', 'round_id']
    cols_for_labels = ['time_to_plan','path_exec_time']

    # if we're learning performance for different planners, we need features of
    # the planners and their configs.  Here we try to extract information about
    # the planner configuration, one-hot encoding strings and bools
    extra_feats = [c for c in df_stats.columns if c.startswith('opt_')]
    _df = df_stats.loc[:,index_cols + extra_feats + cols_for_labels].copy()
    for c in extra_feats:
        if pdt.is_string_dtype(_df[c]) or pdt.is_bool_dtype(_df[c]):
            _df = pd.get_dummies(_df, columns=[c], prefix=c, dtype=int)
        else:
            assert pdt.is_integer_dtype(_df[c]) or pdt.is_float_dtype(_df[c]), f"{c}\n{_df.dtypes}"

    _df['plan_exec_time'] = _df.time_to_plan + _df.path_exec_time
    
    # #################################################################################################
    # ##########     THE SCORING FUNCTIONS !!!   ######################################################
    # #################################################################################################
    agg_cols = [c for c in index_cols if c!='round_id']

    # a combined score of time + the standard deviation
    _sd_plan = _df.groupby(agg_cols)['time_to_plan'].transform('std')
    _sd_exec = _df.groupby(agg_cols)['path_exec_time'].transform('std')
    _df['score'] = _df.plan_exec_time + _sd_plan + _sd_exec

    # calculate the 95th-50th percentile range
    _p50 = _df.groupby(agg_cols).plan_exec_time.transform(lambda x:x.quantile(.50))
    _p95 = _df.groupby(agg_cols).plan_exec_time.transform(lambda x:x.quantile(.95))
    _df['pe_50_95_range'] = _p95 - _p50
    _df['time_plus_tail'] = _df.plan_exec_time + _df.pe_50_95_range

    # combine the features and labels
    df_comb = _df.merge(
        df_feats, how = 'left',
        left_on = ['query_sta', 'query_end'], right_on = ['fr_name', 'to_name']
    )
    df_comb = df_comb.drop(
        columns=['query_sta', 'query_end', 'fr_name', 'to_name'] + cols_for_labels
    )
    return df_comb


def train_test_split(df, seed, test_frac = 0.1):
    """Split the data into train and test, by partitioning the endpoints

    Endpoints are kept completely separate across the training and test sets
    """
    rng = np.random.default_rng(seed=seed)
    ep_idxs = sorted(set(df.fr_idx.unique()).union(df.to_idx.unique()))
    rng.shuffle(ep_idxs)
    cut_at = int(np.ceil(test_frac * len(ep_idxs)))
    print(f"Cutting {len(ep_idxs)} endpoints up to # {cut_at}")
    test_eps = ep_idxs[:cut_at]
    df_test = df.loc[df.fr_idx.isin(test_eps) & df.to_idx.isin(test_eps)]
    df_train = df.loc[~df.fr_idx.isin(test_eps) & ~df.to_idx.isin(test_eps)]
    print(f"Shape of training: {df_train.shape}, test: {df_test.shape}")
    return df_train, df_test


def train_models(train_set, seed=0):
    """Train regression models"""
    X_raw = train_set.drop(columns = DONT_TRAIN_ON).to_numpy()
    
    scaler = StandardScaler().fit(X_raw)
    X = scaler.transform(X_raw)

    rf_params = dict(
        criterion = ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'],
        n_estimators = [50,75,100,125,150],
        # max_depth = list(range(3,50)),
        min_samples_leaf = np.linspace(1e-3,1e-1,10).astype(float).tolist(),
        max_features = ['sqrt', 'log2'],
        max_samples = np.linspace(0.1,0.6,6).round(3).astype(float).tolist(),
    )
    tuning_params = dict(
        n_jobs=-1, random_state=seed, n_iter=24, verbose=2
    )
    
    regressors = {}
    for label_col in PREDICT_COLS:
        reg = Reg(n_jobs=1)
        print(f"Tuning hyperparameters for {label_col}")
        tuned_model = RandomizedSearchCV(reg, rf_params, **tuning_params)
        print("Fitting tuned model")
        regressors[label_col] = tuned_model.fit(X,train_set[label_col])
    return regressors, scaler


def decide_single_bests(train_set):
    """Pick the best planner+config from the training set stats across all
    prediction fields

    """
    sb = {}
    for pred_type in PREDICT_COLS:
        mean_perf = train_set.groupby(['ptype_id','conf_id'])[pred_type].mean()
        best_row = mean_perf.sort_values().reset_index().iloc[0]
        sb[pred_type] = (int(best_row['ptype_id']), int(best_row['conf_id']))
    return sb


def make_predictions(test_set, models, scaler):
    X_raw = test_set.drop(columns = DONT_TRAIN_ON).to_numpy()
    X = scaler.transform(X_raw)
    return {n : models[n].predict(X) for n in PREDICT_COLS}


def evaluate_preds(df_test, preds, single_bests):
    """Evaluate ML predictions against single best and virtual best

    """
    for k in PREDICT_COLS:
        m = maperr(df_test[k].to_numpy(),preds[k])
        r = r2_score(df_test[k].to_numpy(),preds[k])
        print(f"~~~ Predicting {k}. Mean Absolute Percentage Error: {m:.3f}; R2: {r:.3f}")

    # prepare single best and virtual best for comparison later
    _p, _c = single_bests['plan_exec_time']
    cols_to_keep = ['fr_idx', 'to_idx', 'round_id', 'ptype_id', 'conf_id', 'plan_exec_time']
    col_template = ['fr_idx', 'to_idx', 'round_id', 'SRC_planner_id', 'SRC_conf_id', 'SRC_time']

    # What were the times in the test data when the sb was chosen?
    df_sb = (
        df_test.loc[(df_test.ptype_id==_p) & (df_test.conf_id==_c)]
        .reset_index().loc[:,cols_to_keep]
    )
    df_sb.columns = [c.replace('SRC_','sb_') for c in col_template]

    # filter to just the rows with the best performance
    vb_idx = df_test.groupby(cols_to_keep[:3])['plan_exec_time'].idxmin()
    df_vb = df_test.loc[vb_idx].reset_index().loc[:,cols_to_keep]
    df_vb.columns=[c.replace('SRC_','vb_') for c in col_template]

    df_ref = pd.merge(df_sb, df_vb, on=cols_to_keep[:3], how='inner')

    # determine the predicted best planner/config pairing for each query
    # performance ready to compare with SB/VB

    # get the trial identifying cols as a stem
    qry_plan_conf_cols = ['fr_idx', 'to_idx', 'ptype_id', 'conf_id']
    df_pred = df_test.loc[:,qry_plan_conf_cols]

    # bring in the raw ml regression preds (all are duplicated across rounds, as
    # features are the same for each query and planner config)
    df_pred['pred_score'] = preds['score']
    df_pred['pred_time'] = preds['plan_exec_time']
    df_pred['pred_time_plus_tail'] = preds['time_plus_tail']
    df_pred = df_pred.drop_duplicates()
    
    # now choose the best planner and config according to the lowest predicted score
    best_per_query_idx = df_pred.groupby(['fr_idx','to_idx'])['pred_score'].idxmin()
    df_ml_choice = df_pred.loc[best_per_query_idx, qry_plan_conf_cols+['pred_time']]
    df_ml_choice.columns = ['fr_idx', 'to_idx', 'ml_planner_id', 'ml_conf_id', 'ml_reg_pred_time']
    
    # bring in the relevant plan_exec_time
    df_ml = (
        df_ml_choice.merge(
            df_test.loc[:,cols_to_keep],
            left_on = list(df_ml_choice.columns)[:4],
            right_on = qry_plan_conf_cols,
            how = 'inner'
        )
        .rename(columns={'plan_exec_time':'ml_time'})
        .drop(columns=['ptype_id','conf_id'])
    )

    df_eval = pd.merge(df_ref, df_ml, on = cols_to_keep[:3], how='inner')
    print("Eval data:", df_eval, df_eval.describe().T, sep="\n")
    return df_eval

if __name__=='__main__':
    main()
