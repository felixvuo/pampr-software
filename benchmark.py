from klampt.io import resource
from klampt.model import trajectory as traj
from klampt.plan import cspace, motionplanning, robotplanning

import util as ut

import argparse
import copy
import itertools as it
import numpy as np
import pandas as pd
from pathlib import Path
import time
import tqdm


def main():

    ap = argparse.ArgumentParser(description="do some motion planning")
    ap.add_argument('path_to_config', type=Path, help="configuration yaml file")
    ap.add_argument('--plan-endpoints', '-e', type=str, default='0,1',
                    help="Plan between the mth and nth endpoint (comma-separated)")
    ag = ap.add_mutually_exclusive_group()
    ag.add_argument('--run-trial', '-r', action='store_true',
                    help="Run the trial")
    ag.add_argument('--print-commands', '-p', action='store_true',
                    help="Print run commands for parallelisation")
    args = ap.parse_args()

    conf_file = args.path_to_config
    trial_dir = conf_file.parent
    conf, world, robot, space = ut.init_configuration(conf_file)

    if args.run_trial:
        ep_idxs = map(int,args.plan_endpoints.split(','))
        run_trial(conf, world, robot, space, trial_dir, use_endpoints = list(ep_idxs))
    elif args.print_commands:
        print_commands(conf, world, robot, space, trial_dir, conf_file)
    elif args.visualise_configs:
        visualise_configs(conf, world, robot, trial_dir)
    else:
        print("NO ACTION CHOSEN")


def print_commands(conf, world, robot, space, trial_dir, conf_file):
    """Prepare commands for parallelisation, do every combination of endspoints each way"""

    all_endpoints, all_q_names = ut.prepare_named_endpoints(conf, robot, space, trial_dir)
    n = len(all_endpoints)
    for a,b in it.combinations(range(n),r=2):
        for fr,to in ((a,b),(b,a)):
            script_path = Path(__file__)
            conf_rel_path = conf_file.resolve().relative_to(script_path.resolve().parent, walk_up=True)
            print(f"python3 {script_path.name} {conf_rel_path} -r -e {fr},{to}")


def run_trial(conf, world, robot, space, trial_dir, use_endpoints=(0,1)):
    all_endpoints, all_end_names = ut.prepare_named_endpoints(conf, robot, space, trial_dir)
    endpoints = [all_endpoints[x] for x in use_endpoints]
    end_names = [all_end_names[x] for x in use_endpoints]

    all_stats = []
    all_paths = {}

    for p_grp_id, p_grp in enumerate(conf['planners']):
        base_opts = {k:v for (k,v) in p_grp.items() if k!='configs'}
        # base_opts = p_grp.get('base_config',{})
        # base_opts['type'] = p_grp['type']
        planner_configs = p_grp.get('configs', [{}])
        loop_it = it.product(enumerate(planner_configs), range(conf['n_rounds_per_planner']))
        for ((conf_id, extra_conf), round_id) in loop_it:
            opts = copy.deepcopy(base_opts)
            opts.update(extra_conf)

            motionplanning.set_random_seed(round_id)
            planner = robotplanning.MotionPlan(space, **opts)
            planner.setEndpoints(*endpoints)
            print(f"~~~~ STARTING EXPERIMENT.  Planner id {p_grp_id}, "+
                  f"config id {conf_id}, round {round_id}, options: {opts}")
            t0 = time.process_time()
            timed_out = False
            for _ in tqdm.tqdm(range(conf['n_iter_batches'])):
                timeout = conf.get('planning_timeout_sec', -1)
                if (timeout >= 0) and ((time.process_time()-t0) > timeout):
                    timed_out = True
                    break
                planner.planMore(conf['iter_batch_size'])
                if planner.getPath() is not None:
                    break
            t_plan = time.process_time()-t0
            stats = {
                'ptype_id' : p_grp_id, 'conf_id' : conf_id, 'round_id' : round_id,
                'query_sta' : end_names[0], 'query_end' : end_names[1],
                'fr_idx' : use_endpoints[0], 'to_idx' : use_endpoints[1],
                'time_to_plan' : t_plan, 'timed_out' : timed_out,
            }
            stats.update({f'opt_{k}' : v for k,v, in opts.items()})
            stats.update({f'pla_{k}' : v for k,v, in planner.getStats().items()})
            path = planner.getPath()
            if path is not None:
                assert_path_matches_endpoints(path, endpoints)
                print("✅ Got a path with",len(path),"milestones")
                if hasattr(space, 'liftPath'):
                    all_paths[(p_grp_id, conf_id, round_id)] = np.array(space.liftPath(path))
                else:
                    all_paths[(p_grp_id, conf_id, round_id)] = np.array(path)
                stats.update(calculate_path_stats(path, robot))
            else:
                print("❌ No feasible path was found")
            V,E = planner.getRoadmap()
            print(len(V),"feasible milestones sampled,",len(E),"edges connected")
            all_stats.append(stats)
            
        
    flabel = f"{use_endpoints[0]},{use_endpoints[1]}"
    df_stats = pd.DataFrame(all_stats)
    df_stats = df_stats.loc[:,sorted(df_stats.columns)]
    df_stats.to_csv(trial_dir / f"stats-{flabel}.csv", index=False)

    robo_link_names = [robot.link(i).getName() for i in range(robot.numLinks())]
    colnames = ['exp_id', 'conf_id', 'milestone']
    path_dfs = []
    for ((p_grp_id,conf_id,round_id), path) in sorted(all_paths.items()):
        df = pd.DataFrame(path, columns = [f'joint_{n}' for n in robo_link_names])
        df['milestone'] = list(range(len(path)))
        df['ptype_id'] = p_grp_id
        df['conf_id'] = conf_id
        df['round_id'] = round_id
        path_dfs.append(df)
    if len(path_dfs) > 0:
        df_paths = pd.concat(path_dfs, ignore_index=True)
        df_paths.to_csv(trial_dir / f'paths-{flabel}.csv.gz', index=None)
    else:
        print(f"WARNING!  No paths were found between {use_endpoints}")
    print(f"WARNING, running with eps = {space.eps}.  Is this small enough?")


def assert_path_matches_endpoints(path, endpoints):
    args_as_arrays = [np.array(v) for v in (endpoints[0],path[0],endpoints[1],path[-1])]
    np.testing.assert_almost_equal(*args_as_arrays[:2], decimal=3)
    np.testing.assert_almost_equal(*args_as_arrays[2:], decimal=3)
    return True


def calculate_path_stats(path, robot):
    robo_traj = traj.RobotTrajectory(robot, milestones = path)
    full_traj = traj.path_to_trajectory(robo_traj, velocities='minimum-jerk', timing='robot')
    
    stats = {}
    stats['path_n_points'] = len(path)
    stats['path_total_len'] = robo_traj.length()
    stats['path_cost_jerkiness'] = ut.path_jerkiness(path) if len(path) > 3 else ''
    stats['path_cost_angle_sum'] = ut.path_unsmoothness_geometric(path) if len(path) > 3 else ''
    stats['path_exec_time'] = full_traj.duration()
    return stats


    
if __name__=='__main__':
    main()
