import argparse
from klampt.io import resource
from pathlib import Path
import util as ut


ap = argparse.ArgumentParser(description="Edit a robot config")
ap.add_argument('conf_file', type=Path, help="config yaml file")
ap.add_argument('ep_file', type=Path, help="path/to/endpoint.config")
args = ap.parse_args()
conf_file = args.conf_file
conf, world, robot, space = ut.init_configuration(conf_file)
c0 = resource.get(args.ep_file.name, directory = str(args.ep_file.parent))
robot.setConfig(c0)
save,c = resource.edit("Edit config",c0,"Config",world=world)
