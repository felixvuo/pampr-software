from klampt import WorldModel
from klampt.io import resource
from klampt.plan import robotplanning
import util as ut

import argparse
import copy
import itertools as it
import numpy as np
import pandas as pd
from pathlib import Path
import time
import tqdm
import yaml

NON_FEAT_COLS = ['fr_idx','to_idx','fr_name','to_name']

def main():
    ap = argparse.ArgumentParser(
        description="calculate features and labels for robot motion planning experiments"
    )
    ap.add_argument('path_to_config', type=Path, help="configuration yaml file")
    ap.add_argument('--out-feats', '-f', type=Path, default=None, help="save features to this csv file")
    args = ap.parse_args()

    df_feats = extract_features(args.path_to_config)
    if args.out_feats == None:
        print(df_feats.round(2))
    else:
        df_feats.to_csv(args.out_feats, index=None)


    
def extract_features(conf_path):
    conf, world, robot, space = ut.init_configuration(conf_path)
    endpoints, names = ut.prepare_named_endpoints(conf, robot, space, conf_path.parent)
    entries = []
    effector_link = robot.link(robot.numLinks()-1)

    queries = list(it.combinations(range(len(endpoints)),r=2))
    for a,b in tqdm.tqdm(queries):
        for fr,to in ((a,b),(b,a)):
            entry = {'fr_idx' : fr, 'to_idx' : to, 'fr_name' : names[fr], 'to_name' : names[to]}

            # record the actual endpoints themselves in the robot's configuration space
            entry.update(
                {f'rob_fr_{robot.link(d).getName()}' : endpoints[fr][d]
                 for d in range(robot.numLinks())}
            )
            entry.update(
                {f'rob_to_{robot.link(d).getName()}' : endpoints[to][d]
                 for d in range(robot.numLinks())}
            )

            # the "real-world" co-ordinates of the end (effector) link at both endpoints
            robot.setConfig(endpoints[fr])
            eff_pos = effector_link.getWorldPosition([0.0, 0.0, 0.0])
            entry.update({f'eff_fr_{d}' : eff_pos[d] for d in range(3)})
            robot.setConfig(endpoints[to])
            eff_pos = effector_link.getWorldPosition([0.0, 0.0, 0.0])
            entry.update({f'eff_to_{d}' : eff_pos[d] for d in range(3)})

            # minimum distance between any part of the robot and any part of the scene
            # the "self-distance" of the robot at both endpoints
            for link in [robot.link(i) for i in range(robot.numLinks())]:
                l_name = link.getName()
                robot.setConfig(endpoints[fr])
                entry[f'min_dist_fr_{l_name}'] = _calc_min_dists(world, link)
                robot.setConfig(endpoints[to])
                entry[f'min_dist_to_{l_name}'] = _calc_min_dists(world, link)
            entries.append(entry)
    return pd.DataFrame(entries)
    

def _calc_min_dists(world, link):
    distances = [link.geometry().distance_simple(world.terrain(i).geometry())
                 for i in range(world.numTerrains())]
    distances += [link.geometry().distance_simple(world.rigidObject(i).geometry())
                  for i in range(world.numRigidObjects())]
    return np.min(distances)
    

if __name__=='__main__':
    main()
