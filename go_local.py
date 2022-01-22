import argparse
import yaml
import utils.submit_helper as sh
from pathlib import Path
import os
import shutil
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema, TreeMakerSchema
from coffea import processor
import importlib

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-t','--type', help='which type: data, mc', choices=('data', 'mc'), default='mc')
parser.add_argument('-v','--version', help='which nanoAOD version: nanov7, nanov8, nanov9, ..., default: nanov9', default='nanov9')
parser.add_argument('-c', '--channel', help='which channel: vhjj, ssww, your own channel, ...', default='vhjj')
parser.add_argument('-y', '--year', help='which year to run: 2016, 2017, 2018', default='2018')
parser.add_argument('-r', '--redo', help='redo the job, default: False', action='store_true', default=False)
parser.add_argument('--pre', help='the name of the pre-step', default=None)
parser.add_argument('--step', help='which step to run', default='obj_sel')
parser.add_argument('--nworker', help='number of workers, default is 100', default='100')
parser.add_argument('--chunksize', help='chunksize, default is 1000000', default='1000000')
# avoid downloading at the same time
group = parser.add_mutually_exclusive_group()  # type: argparse._MutuallyExclusiveGroup
group.add_argument('-a', '--all', help='run on all, default: True', action='store_false', default=True)
group.add_argument('-s', '--sample', help='run on which sample, can be more than 1', nargs='+')
# ,default=['WpWpJJ_EWK','WpWpJJ_EWK_powheg','WmWmJJ_EWK_powheg'])
args = parser.parse_args()

def get_cfg(args):
    cfg = {
        'year': args.year,
        'channel': args.channel,
        'nano_ver': args.version,
        'sample_type': args.type,
        'all': args.all,
        'sample': args.sample,
        'pre': args.pre,
        'step': args.step,
        'redo': args.redo,
        'nworker': args.nworker,
        'chunksize': args.chunksize,
    }
    with open('./config/dataset_cfg.yaml', 'r') as f:
        ds_cfg = yaml.load(f, Loader=yaml.FullLoader)

    try: 
        nano_verison = ds_cfg[cfg['channel']][cfg['nano_ver']]
    except:
        print("===> Wrong nanoAOD version, terminate!!!")
        exit(1)
    else:
        pass

    with open('./config/condor_cfg.yaml', 'r') as f:
        condor_cfg = yaml.load(f, Loader=yaml.FullLoader)
    in_dir = condor_cfg['in_dir']  # input area
    out_dir = condor_cfg['out_dir']  # output ntuple area

    with open('./config/step_cfg.yaml', 'r') as f:
        step_cfg = yaml.load(f, Loader=yaml.FullLoader)
        step_handle = step_cfg[cfg['channel']][cfg['step']]
    with open(f"./datasets/{ds_cfg[cfg['channel']][cfg['nano_ver']][cfg['sample_type']][int(cfg['year'])]}", 'r') as f:
        ds_yml = yaml.load(f, Loader=yaml.FullLoader)
    
    cfg['in_dir'] = in_dir
    cfg['out_dir'] = out_dir
    cfg['ds_yml'] = ds_yml
    cfg['step_handle'] = step_handle
    return cfg


def get_file_info(fpath,sample_name):
    flist = []
    for path in Path(fpath).rglob('*.root'):
        # print(path.resolve())
        flist.append(str(path.resolve()))
    fdict = {sample_name: flist}
    return fdict

def start_run(file_dict, cfg):
    if cfg['step'] == 'obj_sel':
        the_processor = importlib.import_module(f"{cfg['step_handle']}.obj_sel")
        results = processor.run_uproot_job(
            file_dict,
            treename='Events',
            processor_instance=the_processor(cfg['out_dir'],cfg['year'],cfg['sample_type']=="data"),
            executor=processor.futures_executor,
            executor_args={"schema": NanoAODSchema, "workers": int(cfg['nworker'])},
            chunksize=int(cfg['chunksize']),
            maxchunks=None,
        )    
    return True

def handle_run(cfg, sample=None):
    print("===> submit_command", "year:", cfg['year'], "sample:", sample)

    # sample dataset name
    sname = cfg['ds_yml'][sample]['dataset']
    if cfg['pre'] == None:
        if sname.startswith("gsiftp://") or sname.startswith("local:"):
            if sname.endswith("/"):
                sname = sname.split("/")[-2]
            else:
                sname = sname.split("/")[-1]
            in_path = f"{cfg['in_dir']}/{cfg['nano_ver']}/private/{sname}/"
        else:
            in_path = f"{cfg['in_dir']}/{cfg['nano_ver']}/{sname}/"
    else:
        in_path = f"{cfg['out_dir']}/{cfg['nano_ver']}/{cfg['pre']}/{sample}/"

    out_path = f"{cfg['out_dir']}/{cfg['nano_ver']}/{cfg['step']}/{sample}/"
    if not os.path.exists(in_path):
        print("xxx","Invalid input path:",in_path, "please check!!!")
        exit(0)
    if os.path.exists(out_path):
        if cfg['redo']:
            shutil.rmtree(out_path)
            print(f"[  Redo  ] {sample} path: {out_path} is removed!!!")
        else:
            print(f"[  Out   ] {sample} path: {out_path} is already there, skip!!!")
            return True
    else:
        os.makedirs(out_path)

    file_dict = get_file_info(in_path,sample)
    print("---", "sample:", sample, ", #file:", len(file_dict[sample]))
        
    start_run(file_dict, cfg)
    print("<=== submit_command END :)")
    return True


def to_skim(year, sample_type):
    print("===> to_submit:", "year:", year, "sample_type:", sample_type)
    cfg = get_cfg(args)

    if cfg['sample']:
        for isp in cfg['sample']:
            if isp in cfg['ds_yml']:
                handle_run(cfg, isp)
            else:
                print("--- submit info:", "no sample: ", isp, "in", year)
    else:
        for isp in cfg['ds_yml']:
            handle_run(cfg, isp)
    print("<=== to_submit END :)")


if __name__ == '__main__':
    to_skim(args.year, args.type)
