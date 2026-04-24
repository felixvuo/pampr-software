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
    ap = argparse.ArgumentParser(description = "Show a query between two robot poses")
    ap.add_argument('conf_file', type=Path, help="Location of trial config file")
    ap.add_argument('source_config_path', type=Path, help="Starting .config file")
    ap.add_argument('dest_config_path', type=Path, help="Goal .config file")
    ap.add_argument('--thumbs_dir', '-t', type=Path, default=None,
                    help="Save thumbnails to this directory (default: don't save)")
    args = ap.parse_args()
    
    trial_dir = args.conf_file.parent
    show_query(
        args.conf_file,
        args.source_config_path,
        args.dest_config_path,
        save_dir = args.thumbs_dir,
    )


def show_query(conf_file, start_path, end_path, save_dir=None):
    conf, world, robot, space = ut.init_configuration(conf_file)
    
    start_cfg = resource.get(start_path.name, directory=str(start_path.parent))
    end_cfg = resource.get(end_path.name, directory=str(end_path.parent))

    for i in range(world.numTerrains()):
        vis.add(f"terrain_{i}", world.terrain(i))
    robot.setConfig(start_cfg)
    vis.add("robot_start", robot, hide_label=True)
    vis.setColor("robot_start", 0.2, 0.5, 1.0, 1.0)
    vis.add("robot_end", end_cfg, robot=robot, hide_label=True)
    vis.setColor("robot_end", 0.1, 0.8, 0.1, 0.5) 
    
    
    # Setup Camera
    vis.show(True)
    vp = kvv.getViewport()
    camera_file = conf_file.parent / 'camera-viewport.txt'
    if camera_file.exists():
        vp.load_file(str(camera_file))
        kvv.setViewport(vp)
    
    vis.setWindowTitle(f"Comparison: {start_path.stem} vs {end_path.stem}")
    vis.update()
    
    # Allow a moment for the renderer to catch up before screenshot
    time.sleep(0.5) 

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        img_path = save_dir / f"compare_{start_path.stem}_{end_path.stem}.png"
        ss = vis.screenshot('Image', False)
        if ss:
            ss.save(str(img_path), 'PNG')
            print(f"Saved comparison to {img_path}")

    vis.spin(float('inf'))
    vis.show(False)

if __name__=='__main__':
    main()

