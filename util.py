from klampt import WorldModel
from klampt.io import resource
from klampt.math import vectorops
from klampt.model import ik
from klampt.plan import robotplanning

import numpy as np
from pathlib import Path
import yaml

def init_configuration(conf_file):
    trial_dir = conf_file.parent
    with open(conf_file, 'rt') as f:
        conf = yaml.safe_load(f)
    world = WorldModel()
    world.readFile(str(trial_dir / conf['world_file']))
    world.loadElement(str(trial_dir / conf['robot_file']))
    robot = world.robot(0)
    space = robotplanning.make_space(world,robot,edgeCheckResolution=conf.get('eps',0.5))

    return conf, world, robot, space


def prepare_named_endpoints(conf, robot, space, trial_dir):
    """Load endpoints from saved configs"""
    indiv_end_keys = ['start_config_file','goal_config_file']
    collection_keys = ['robot_configs_dir', 'robot_configs_glob']
    if len(set(conf.keys()).intersection(set(indiv_end_keys))) == 2:
        ep_paths = [trial_dir / conf[k] for k in indiv_end_keys]
        names = [p.name for p in ep_paths]
        points = [resource.get(p.name, directory = str(p.parent)) for p in _ep_paths]
    elif len(set(conf.keys()).intersection(set(collection_keys))) == 2:
        ep_dir = trial_dir / conf.get('robot_configs_dir', '.')
        pattern = conf.get('robot_configs_glob', '*config')
        names = [c.name for c in sorted(ep_dir.glob(pattern))]
        points = [resource.get(n, directory=str(ep_dir)) for n in names]
    if np.array(points).shape[1]==robot.numLinks():
        endpoints = points
    else:
        endpoints = [space.project(p) for p in points]
    assert len(endpoints) == len(names)
    for q,n in zip(endpoints,names):
        assert space.feasible(q), f"Query from {n} is not feasible"
    return endpoints, names


def path_unsmoothness_geometric(path):
    """Deviation from straight line (sum of angles between triples)"""

    assert len(path)>2, "Can't do smoothness (angles) with fewer than 3 points"
    a_path = np.array(path)

    # vectors between adjacent points
    a = a_path[1:-1] - a_path[:-2]
    b = a_path[2:] - a_path[1:-1]

    dot_prod = np.sum(a*b, axis=1)
    norms_a = np.linalg.norm(a, axis=1)
    norms_b = np.linalg.norm(b, axis=1)

    # a tiny extra to avoid dividing by zero
    epsilon = 1e-10

    # work out cosine theta, but keep in [-1,1] in case floats multiply outside
    cosines = np.clip(dot_prod / (norms_a*norms_b + epsilon), -1.0, 1.0)
    angles = np.arccos(cosines)
    sum_of_deviations = float(np.sum(np.pi - angles))

    return sum_of_deviations


def path_jerkiness(path):
    """Unsmoothness/jerkiness of a path based on integrating squared jerk"""
    assert len(path)>3, "Can't do smoothness (3rd diff) with fewer than 4 points"
    a_path = np.array(path)

    # approximate 3rd derivative
    diff3 = np.diff(a_path, n=3, axis=0)

    # sum of squares of x'''
    squared_norm_sum = np.sum(diff3 ** 2, axis=1)
    smoothness_metric = float(np.sum(squared_norm_sum))

    return smoothness_metric



