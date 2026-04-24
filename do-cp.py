import argparse
import itertools as it
import numpy as np
from ortools.sat.python import cp_model
import pandas as pd
import pandas.api.types as pdt
from pathlib import Path
import pickle
from sklearn.ensemble import RandomForestRegressor as Reg
#from sklearn.ensemble import ExtraTreesRegressor as Reg
from sklearn.metrics import mean_absolute_percentage_error as maperr
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import time
import tqdm

import util as ut

def main():
    ap = argparse.ArgumentParser(
        description="train (and evaluate) model"
    )
    ap.add_argument('path_to_conf', type=Path, default=None,
                    help="where's the trial config yaml file?")
    ap.add_argument('--solve', '-s', action='store_true', default=False,
                    help="Attempt to solve with or-tools")
    ap.add_argument('--eprime-params', '-p', action='store_true', default=False,
                    help="Produce eprime parameter files for Savile Row")
    ap.add_argument('--evaluate', '-e', action='store_true', default=False,
                    help="Work out resulting time from predictions alongside VB and SB")
    args = ap.parse_args()

    conf, _, _, _ = ut.init_configuration(args.path_to_conf)
    trial_dir = args.path_to_conf.parent
    granularity = conf['cp_granularity']
    pred_time_field = conf.get('cp_pred_time_field','score')
    setup_name = conf.get('setup_name', 'default')
    solutions = []

    df_all_preds = pd.read_csv(trial_dir / "preds.csv.gz")
    if args.solve or args.eprime_params:
        for seed, df in df_all_preds.groupby('seed'):
            rng = np.random.default_rng(seed=seed)
            places = sorted(set(df.fr_idx.unique().tolist()).union(df.to_idx.unique().tolist()))
            df_planner_configs = df.loc[:,['ptype_id','conf_id']].drop_duplicates()
            pred_times = []
            for (pid,cid), df2 in df.groupby(['ptype_id', 'conf_id']):
                times = np.zeros(shape=(len(places),len(places)),dtype=np.int32)
                for _idx, row in df2.iterrows():
                    scaled_time = int(np.ceil(row[pred_time_field] * granularity))
                    times[places.index(row.fr_idx), places.index(row.to_idx)] = scaled_time
                pred_times.append(times.tolist())
            params = dict(
                n_places = len(places),
                n_algos = df_planner_configs.shape[0],
                switch_pen = granularity * conf['cp_switch_penalty_sec'],
                times = pred_times,
                home_idx = int(rng.choice(range(len(places))))
            )
            if args.eprime_params:
                eprime_param_lines = [f"letting {k} be {v}" for k,v in params.items()]
                with open(trial_dir / f"plan-preds-{seed:04d}.param", 'wt') as f:
                    f.write('\n'.join(eprime_param_lines)+'\n')
            if args.solve:
                _t0 = time.process_time()
                result = solve_multimodal_tsp(**params)
                _cp_sol_time = time.process_time() - _t0
                print(f"For seed {seed}, OR-tools took {_cp_sol_time}s and got:\n{result}")
                solutions.append(
                    list(it.chain(
                        [seed],
                        result['tour'],
                        result['algos'],
                        [result['obj_val_time_units']//granularity]
                    ))
                )
    if args.solve:
        arr_sol = np.array(solutions, dtype=int)
        # the width of each row is 1 + (n_steps+1) + (n_steps) + 1
        n_steps = (arr_sol.shape[1] - 3) // 2
        _cols = list(it.chain(
            ['seed'],
            [f'tour_{i:03d}' for i in range(n_steps+1)],
            [f'algo_{i:03d}' for i in range(n_steps)],
            ['time']
        ))
        df = pd.DataFrame(arr_sol, columns=_cols, dtype='Int32')
        df.to_csv(f'routes-{setup_name}.csv', index=None)
    if args.evaluate:
        do_evaluation(conf, trial_dir)


def do_evaluation(conf, trial_dir):
    # we have r rounds of results, r^(n. of test endpoints) combinations;
    # let's sample for evaluation
    # 
    # - work out the timings of using the recommended sequence of algos
    # - solve the TSP with the actual exact timings available
    # - solve the TSP with the single best planner, and a linear regression
    #   linking features to plan/exec time
    setup_name = conf.get('setup_name', 'default')
    df_all_routes = pd.read_csv(trial_dir / f"routes-{setup_name}.csv")
    df_all_stats = pd.read_csv(trial_dir / "stats.csv.gz")
    df_all_preds = pd.read_csv(trial_dir / "preds.csv.gz")
    df_single_best = pd.read_csv(trial_dir / "single-bests.csv")
    cp_gran = conf['cp_granularity']
    t_pen = conf['planning_timeout_sec'] * conf['eval_timeout_penalty']

    eval_entries = []
    
    # loop through all the train/test splits
    for split, df_preds in df_all_preds.groupby('seed'):
        print(f"Starting on split #{split}")

        # all the planner/config pairings
        all_algos = sorted(
            df_all_stats.loc[:,['ptype_id','conf_id']].drop_duplicates().to_numpy().tolist()
        )
        
        # to figure out the places, we need to find all the places involved in
        # our split, order them and index into them
        df = df_preds.loc[:,['fr_idx','to_idx']].drop_duplicates()
        places = sorted(map(int,set(df.to_numpy().flatten())))
        rng = np.random.default_rng(seed=split)
        home_idx = rng.choice(range(len(places)))

        # we just need the stats for the test set
        df_stats_test = df_all_stats.loc[
            df_all_stats.fr_idx.isin(places) & df_all_stats.to_idx.isin(places)
        ]

        # collect the routes according to the different "selectors"
        route_info = {}

        # first look up the CP+ML recommended route/algos
        df_route = df_all_routes.loc[df_all_routes.seed==split]
        assert df_route.shape[0] == 1
        tour_idxs = df_route.iloc[0][[c for c in df_route.columns if c.startswith('tour_')]]
        algo_idxs = df_route.iloc[0][[c for c in df_route.columns if c.startswith('algo_')]]
        
        route_info['ml+cp'] = {
            'tour_places' : [places[i] for i in tour_idxs],
            'route_algos' : [tuple(all_algos[a]) for a in algo_idxs]
        }

        # now let's do VB...
        # we need new route and algos, worked out via tsp
        oracle_times = []
        for (pid,cid), df_pla_con in df_stats_test.groupby(['ptype_id', 'conf_id']):
            times = np.zeros(shape=(len(places),len(places)),dtype=np.int32)
            cols = ['fr_idx','to_idx','path_exec_time','time_to_plan']
            df_avg = df_pla_con.loc[:,cols].groupby(['fr_idx','to_idx']).median().reset_index()
            for row in df_avg.itertuples():
                if any([np.isnan(val) for val in row]):
                    tot_time = t_pen
                else:
                    tot_time = row.path_exec_time + row.time_to_plan
                scaled_time = int(np.ceil(tot_time * cp_gran))
                times[places.index(row.fr_idx), places.index(row.to_idx)] = scaled_time
            oracle_times.append(times.tolist())
        print("Solving for oracle")
        _t0 = time.process_time()
        cp_result = solve_multimodal_tsp(
            n_places = len(places),
            n_algos = len(all_algos),
            times = oracle_times,
            switch_pen = conf['cp_switch_penalty_sec'] * cp_gran,
            home_idx = home_idx
        )
        print(f"VB route solving took {time.process_time()-_t0}s.")
        route_info['vb_random'] = {
            'tour_places' : [places[p] for p in cp_result['tour']],
            'route_algos' : [tuple(all_algos[a]) for a in cp_result['algos']]
        }
        route_info['vb_lucky'] = route_info['vb_random']

        # and finally Single Best
        sb_pla_con = df_single_best.query(f"measure=='plan_exec_time' and split=={split}")
        assert sb_pla_con.shape == (1,4), "Was expecting a single row to match"
        sb_p = sb_pla_con.iloc[0]['ptype_id']
        sb_c = sb_pla_con.iloc[0]['conf_id']
        sb_times = np.zeros(shape=(len(places),len(places)),dtype=np.int32)
        q = f"ptype_id=={sb_p} and conf_id=={sb_c}"

        # work out the best route using the times predicted by the ML regression
        for _idx, row in df_preds.query(q).iterrows():
            scaled_time = int(np.ceil(row.plan_exec_time * cp_gran))
            sb_times[places.index(row.fr_idx), places.index(row.to_idx)] = scaled_time

        print("Solving for single best")
        _t0 = time.process_time()
        cp_result = solve_multimodal_tsp(
            n_places = len(places),
            n_algos = 1,
            times = [sb_times.tolist()],
            switch_pen = conf['cp_switch_penalty_sec'] * conf['cp_granularity'],
            home_idx = home_idx
        )
        print(f"SB route solving took {time.process_time()-_t0}s.")
        route_info['sb_regress'] = {
            'tour_places' : [places[p] for p in cp_result['tour']],
            'route_algos' : [tuple(all_algos[a]) for a in cp_result['algos']]            
        }


        # All the routes are ready, we begin simulating a real run by sampling
        # from ground truth results
        print(f"   Sampling")
        _debug_times = []
        for sample_id, selector in tqdm.tqdm(
                it.product(range(conf['eval_n_samples']), route_info.keys()),
                total = conf['eval_n_samples']*len(route_info.keys()),
        ):
            route_algos = route_info[selector]['route_algos']
            tour_places = route_info[selector]['tour_places']
            route_times = []

            assert (len(route_algos)+1) == len(tour_places), "Should be (n_places-1) steps"

            switches = 0
            timeouts = 0
            total_time = 0
            for step, algo in enumerate(route_algos):
                src = tour_places[step]
                dst = tour_places[step+1]
                q = f"ptype_id=={algo[0]} and conf_id=={algo[1]} and fr_idx=={src} and to_idx=={dst}"

                # sample one of the "rounds"
                cols = ['path_exec_time','time_to_plan','timed_out']
                if selector.endswith('_lucky'):
                    # we're cherry-picking the very best time (for the 'lucky' virtual best)
                    _df = df_all_stats.query(q).copy()
                    _df['total_pe_time'] = _df.time_to_plan + _df.path_exec_time
                    row = _df.sort_values('total_pe_time').loc[:,cols].iloc[0]
                else:
                    row = df_all_stats.query(q).sample(1).loc[:,cols].iloc[0]
                timed_out = (row['timed_out']==True) or (row.isna().any())
                timeouts += 1 if timed_out else 0
                switches += 1 if ((step == 0) or (route_algos[step-1] != route_algos[step])) else 0
                step_time = t_pen if timed_out else (row['path_exec_time'] + row['time_to_plan'])
                total_time += step_time
                route_times.append(step_time)
            eval_entries.append(dict(
                time = total_time + conf['cp_switch_penalty_sec'] * switches,
                n_timeouts = timeouts,
                selector = selector,
                split = split,
                sample = sample_id,
            ))
            _debug_times.append(route_times+[selector])
        pd.DataFrame(_debug_times).to_csv(f"split-{split}-times.csv", index=None)
            
    df_eval = pd.DataFrame(eval_entries)
    df_eval.to_csv(trial_dir / f'eval-{setup_name}.csv.gz', index=False)
    print(
        df_eval, df_eval.info(),
        df_eval.groupby('selector')[['time','n_timeouts']].describe().round(2),
        sep="\n"
    )
            


    
def solve_multimodal_tsp(n_places, n_algos, times, switch_pen, home_idx):
    print(f"DEBUG..... starting solve_multimodal_tsp with:\n{np.array(times).shape}")
    model = cp_model.CpModel()
    n_steps = n_places
    
    # Convert times to a numpy array for cleaner processing
    times_arr = np.array(times)
    
    # Decision Variables
    tour = [model.NewIntVar(0, n_places, f'tour_{i}') for i in range(n_places+1)]
    algo = [model.NewIntVar(0, n_algos - 1, f'algo_{t}') for t in range(n_steps)]
    switch = [model.NewBoolVar(f'switch_{t}') for t in range(n_steps)]
    max_time = int(times_arr.max())
    time_used = [model.NewIntVar(0, max_time, f'time_{t}') for t in range(n_steps)]

    # constraints


    # start and end at home, otherwise a circuit...
    model.AddAllDifferent(tour[:-1])
    model.Add(tour[0]==home_idx)
    model.Add(tour[-1]==home_idx)

    # keep track of algorithm switching
    model.Add(switch[0] == 1) 
    for t in range(1, n_steps):
        model.Add(algo[t-1] == algo[t]).OnlyEnforceIf(switch[t].Not())
        model.Add(algo[t-1] != algo[t]).OnlyEnforceIf(switch[t])

    # hook up the times matrix as a table constraint
    allowed_transitions = [
        (int(a), int(p1), int(p2), int(times_arr[a, p1, p2]))
        for a, p1, p2 in np.ndindex(times_arr.shape)
    ]

    for t in range(n_steps):
        model.AddAllowedAssignments(
            [algo[t], tour[t], tour[t+1], time_used[t]], 
            allowed_transitions
        )

    # objective function
    travel_cost = sum(time_used)
    switch_cost = sum(s * switch_pen for s in switch)
    model.Minimize(travel_cost + switch_cost)

    # solve and collect results
    solver = cp_model.CpSolver()
    # solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "tour": [solver.Value(t) for t in tour],
            "algos": [solver.Value(a) for a in algo],
            "switch": [solver.Value(s) for s in switch],
            "time_used" : [solver.Value(t) for t in time_used],
            "obj_val_time_units": int(solver.ObjectiveValue())
        }
    return None

            
if __name__=='__main__':
    main()
