import argparse
from klampt import vis
from klampt.io import resource
from klampt.vis import visualization as kvv
import numpy as np
from pathlib import Path
import time
import tqdm
import util as ut


def main():
    ap = argparse.ArgumentParser(description = "Show a robot pose")
    ap.add_argument('conf_file', type=Path, help="Location of trial config file")
    ap.add_argument('ep_dir', type=Path, help="Location of endpoints (*.config)")    
    ap.add_argument('--thumbs_dir', '-t', type=Path, default=None,
                    help="Save thumbnails to this directory (default: don't save)")
    ap.add_argument('--show-length', '-s', type=float, default=1.0,
                    help="Show each position for this many seconds")
    args = ap.parse_args()
    
    trial_dir = args.conf_file.parent
    config_paths = sorted(args.ep_dir.glob("*.config"))
    show_configs(
        args.conf_file,
        config_paths,
        save_dir = args.thumbs_dir,
        show_for = args.show_length
    )

    
def show_configs(conf_file, paths, save_dir=None, show_for=1.0):
    conf, world, robot, space = ut.init_configuration(conf_file)
    vis.add("world", world)
    vis.show(True)
    vp = kvv.getViewport()
    vp.load_file(str(conf_file.parent / 'camera-viewport.txt'))
    kvv.setViewport(vp)
    coords = []
    for path in paths:
        print(f"Showing {path}")
        c = resource.get(path.name, directory = str(path.parent))
        robot.setConfig(c)
        coords.append(robot.link(robot.numLinks()-1).getWorldPosition( (0,0,0) ))
        vis.setWindowTitle(str(path))
        vis.update()
        time.sleep(show_for)
        if save_dir != None:
            save_dir.mkdir(parents=True, exist_ok=True)
            img_path = save_dir / f"{path.stem}.png"
            ss = vis.screenshot('Image',False)
            ss.save(str(img_path),'PNG')
            print(f"Saved thumbnail for {img_path.name}")
    vis.spin(float('inf'))
    vis.show(False)
    a_coords = np.array(coords)
    print(f"Last link co-ordinates:\n{a_coords}")

if __name__=='__main__':
    main()

