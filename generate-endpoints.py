import argparse
from klampt.io import resource
from klampt.math import vectorops
from klampt.model import collide, ik
import numpy as np
from pathlib import Path
import tqdm
import util as ut

def main():
    ap = argparse.ArgumentParser(
        description = "Generate new endpoints within given world and boubnds")
    ap.add_argument('conf_file', type=Path, help="Location of config file")
    ap.add_argument('limits', type=str,
                    help="cartesian bounds of end effector as `x_lo,x_hi,y_lo,y_hi,z_lo,z_hi`")
    ap.add_argument('out_dir', type=Path, help="Save configs to this directory")
    ap.add_argument('--n-wanted', '-n', type=int, default=1, help="how many to generate")
    ap.add_argument('--max-tries', '-t', type=int, default=1000, help="how many attempts to make")
    ap.add_argument('--min-dist', '-m', type=float, default=1.0,
                    help="initial minimum distance in cartesian space from other generated points")
    ap.add_argument('--min-dist-decay', '-d', type=float, default=0.1,
                    help="decay in allowed inter-point distance")
    ap.add_argument('--seed', '-s', type=int, default=0, help="random seed")
    args = ap.parse_args()
    
    conf, world, robot, space = ut.init_configuration(args.conf_file)
    hand_link = robot.link(conf['robot_link_to_position'])
    extremes = np.array(
        [float(x.strip()) for x in args.limits.split(',')]
    ).reshape( (3,2) ) # bounds for each dimension
    print(f"Parsed extremes by dimension x, y, z: \n{extremes}")

    _configs = []
    _coords = []
    threshold = args.min_dist
    rng = np.random.default_rng(seed=0)
    for attempt in tqdm.tqdm(range(args.max_tries)):
        target_pos = [rng.uniform(*extremes[axis]) for axis in range(3)]
        robot.setConfig(space.sample())
        obj = ik.objective(hand_link, local=[0,0,0], world=target_pos)
        if ik.solve(obj, iters=100, tol=1e-3):
            c = robot.getConfig()
            coo = hand_link.getWorldPosition((0.0,0.0,0.0))
            if is_feasible(c, world, robot, space) and is_novel(coo, _coords, threshold):
                _configs.append(c)
                _coords.append(coo)
                threshold *= (1-args.min_dist_decay)
                file_path = args.out_dir / f"gen_{len(_configs)-1:03d}.config"
                resource.set(str(file_path.name), c, directory = str(file_path.parent))
                print(f"Made endpoint, threshold now {threshold}")
        if len(_configs) >= args.n_wanted:
            break
    print(f"Created {len(_configs)} endpoints")

def is_feasible(q, world, robot, space):
    """Checks for both environment and self-collisions."""
    if not space.feasible(q):
        return False
    robot.setConfig(q)
    w_col = collide.WorldCollider(world)
    if any(w_col.collisions()):
        return False
    return True

def is_novel(coo, existing_coords, threshold):
    """Checks if the configuration is far enough from all others in the library."""
    for other_coo in existing_coords:
        if vectorops.distance(coo, other_coo) < threshold:
            return False
    return True


if __name__=='__main__':
    main()
