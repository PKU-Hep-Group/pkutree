import glob
from heapq import merge
import os
import os.path
import yaml
from pathlib import Path
import shutil
from coffea.nanoevents import NanoAODSchema, BaseSchema
from coffea import processor
import utils.common_helper as hcom
import utils.step_helper as hstep
import time
import pickle
import awkward as ak
import numpy as np

import logging
logger = logging.getLogger('run_helper')

def get_cfg(args):
    cfg = {
        'year': args.year,
        'channel': args.channel,
        'data': args.data,
        'version': args.version,
        'redo': args.redo,
        'sample': args.sample,
        'all': args.all,
        'format': args.format,
        'norm': args.norm,
        'merge': args.merge,
        'pre': args.pre,
        'step': args.step,
        'nworker': args.nworker,
        'chunksize': args.chunksize,
        'schema': args.schema,
        'executor': args.executor,
    }
    with open(hcom.abs_path('config/dataset_cfg.yaml'), 'r') as f:
        ds_cfg = yaml.load(f, Loader=yaml.FullLoader)
    try: 
        ds_cfg[cfg['channel']][cfg['version']]
    except:
        logger.error("[Error] Invalid nanoAOD version '%s', terminate now", ds_cfg[cfg['channel']][cfg['version']])
        exit(1)


    with open(hcom.abs_path('config/path_cfg.yaml'), 'r') as f:
        condor_cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg['in_dir'] = condor_cfg['in_dir']  # input area
    cfg['out_dir'] = condor_cfg['out_dir']  # output ntuple area
    cfg['job_dir'] = condor_cfg['job_dir']
    
    sample_type = "mc"
    if cfg['data']:
        sample_type = "data"
    with open(hcom.abs_path(f"datasets/{ds_cfg[cfg['channel']][cfg['version']][sample_type][int(cfg['year'])]}"), 'r') as f:
        ds_yml = yaml.load(f, Loader=yaml.FullLoader)
    cfg['ds_yml'] = ds_yml

    logger.info(
        """
--------------------------------
>>> Following is the options <<<
         'year' : %s,
      'channel' : %s,
         'data' : %s,
      'version' : %s,
         'redo' : %s,
       'sample' : %s,
          'all' : %s,
       'format' : %s,
         'norm' : %s,
        'merge' : %s,
          'pre' : %s,
         'step' : %s,
      'nworker' : %s,
    'chunksize' : %s,
       'schema' : %s,
     'executor' : %s,
--------------------------------
        """,
        args.year,
        args.channel,
        args.data,
        args.version,
        args.redo,
        args.sample,
        args.all,
        args.format,
        args.norm,
        args.merge,
        args.pre,
        args.step,
        args.nworker,
        args.chunksize,
        args.schema,
        args.executor,
    )
    return cfg

def get_file_list(fpath, pattern):
    flist = []
    for path in Path(fpath).rglob(pattern):
        # print(path.resolve())
        flist.append(str(path.resolve()))
    return flist

def get_sample_info(sample, cfg):
    # get xsec_norm
    norm = 1.0
    if cfg['norm'] and not cfg['data']: 
        sxsec = cfg['ds_yml'][sample]['xsec']
        with open(hcom.abs_path(f"rundoc/{cfg['channel']}_{cfg['year']}_{cfg['version']}_obj_sel_count.yaml"), 'r') as f:
            ncount_cfg = yaml.load(f, Loader=yaml.FullLoader)
            if not sample in ncount_cfg.keys():
                logger.error("[Error] sample '%s' not found in ncount_cfg", sample)
                exit(1)
            else:
                sneff = float(ncount_cfg[sample]['neff'])
                xsec_norm = sxsec * 1000 / sneff
        with open(hcom.abs_path("config/lumi_cfg.yaml"), 'r') as f:
            lumi_cfg = yaml.load(f, Loader=yaml.FullLoader)
            lumi = lumi_cfg[cfg['year']]
        norm = xsec_norm * lumi

    # sample dataset name
    sdataset = cfg['ds_yml'][sample]['dataset']
    if cfg['pre'] == None:
        if sdataset.startswith("gsiftp://") or sdataset.startswith("local:"):
            if sdataset.endswith("/"):
                sdataset = sdataset.split("/")[-2]
            else:
                sdataset = sdataset.split("/")[-1]
            in_path = f"{cfg['in_dir']}/{cfg['version']}/private/{sdataset}/"
        else:
            in_path = f"{cfg['in_dir']}/{cfg['version']}/{sdataset}/"
    else:
        in_path = f"{cfg['out_dir']}/{cfg['version']}/{cfg['pre']}/{cfg['year']}/{sample}/"

    out_path = f"{cfg['out_dir']}/{cfg['version']}/{cfg['step']}/{cfg['year']}/{sample}/"
    
    done_before = False
    try:
        with open(hcom.abs_path(f"rundoc/{cfg['channel']}_{cfg['year']}_{cfg['version']}_{cfg['step']}_count.yaml"), 'r') as f:
            ncount_cfg = yaml.load(f, Loader=yaml.FullLoader)
            if sample in ncount_cfg.keys():
                done_before = True
    except:
        pass

    if not os.path.exists(in_path):
        logger.err("Invalid input path %s, exit now",in_path)
        exit(1)
    if os.path.exists(out_path):
        if cfg['redo']:
            shutil.rmtree(out_path)
            logger.warning("[redo] %s (%s) is removed",sample,out_path)
        elif done_before:
            logger.info("%s (%s) is already done, skip",sample,out_path)
            return
        else:
            shutil.rmtree(out_path)
            logger.warning("[redo] %s (%s) is not finished before, automatically redo",sample,out_path)
        os.makedirs(out_path)
    else:
        os.makedirs(out_path)

    if cfg['norm']: 
        sinfo = {
            "treename": "Events",
            "files": get_file_list(in_path,f"*.{cfg['format']}"),
            "metadata": {
                "outpath": out_path,
                "norm": norm,
            }
        }
    else:
        sinfo = {
            "treename": "Events",
            "files": get_file_list(in_path,f"*.{cfg['format']}"),
            "metadata": {
                "outpath": out_path,
            }
        }

    return sinfo

def get_file_dict(cfg):
    # filelist = {
    #     "DummyBad": {
    #         "treename": "Events",
    #         "files": [osp.abspath("tests/samples/non_existent.root")],
    #     },
    #     "ZJets": {
    #         "treename": "Events",
    #         "files": [osp.abspath("tests/samples/nano_dy.root")],
    #         "metadata": {"checkusermeta": True, "someusermeta": "hello"},
    #     },
    #     "Data": {
    #         "treename": "Events",
    #         "files": [osp.abspath("tests/samples/nano_dimuon.root")],
    #         "metadata": {"checkusermeta": True, "someusermeta2": "world"},
    #     },
    # }
    fdict = {}
    if cfg['sample']:
        for isp in cfg['sample']:
            if isp in cfg['ds_yml'].keys():
                sinfo = get_sample_info(isp, cfg)
                if sinfo:
                    if len(sinfo) > 0:
                        fdict[isp] = sinfo
                else:
                    pass
    elif cfg['all']:
        for isp in cfg['ds_yml'].keys():
            sinfo = get_sample_info(isp, cfg)
            if sinfo:
                if len(sinfo) > 0:
                    fdict[isp] = sinfo
            else:
                pass
    # print(fdict)
    return fdict


def running(fdict, cfg):

    schema_dict = {
        'NanoAODSchema': NanoAODSchema,
        'BaseSchema': BaseSchema,
    }

    executor_dict = {
        'FuturesExecutor': processor.FuturesExecutor(workers=int(cfg['nworker'])), # 
        'IterativeExecutor': processor.IterativeExecutor(),
    }
    if cfg['executor'] == 'DaskExecutor':
        from dask.distributed import Client
        client = Client(n_workers=int(cfg['nworker']), threads_per_worker=1, memory_limit='4GB')
        executor_dict['DaskExecutor'] = processor.DaskExecutor(client)

    logger.info(">>> Start >>> Schema: %s, Executor: %s",cfg['schema'],cfg['executor'])
    run = processor.Runner(
        executor=executor_dict[cfg['executor']],
        schema=schema_dict[cfg['schema']],
        format=cfg['format'],
        chunksize=float(cfg['chunksize']),
    )
    
    fdict_new, the_processor = hstep.get_step_module(fdict, cfg)
    results = run(
        fdict_new,
        "Events",
        processor_instance= the_processor
    )

    return results

def merge_parquet(file_folder,match_pattern="*.parquet",chunksize=300000,outprefix="merged"):
    # logger.info("[merge_parquet] file_folder: %s",file_folder)
    toc = time.monotonic()
    print("[merge_parquet] file_folder: %s" % (file_folder))
    file_list = get_file_list(file_folder,match_pattern)
    total_entry = 0
    file_count = 0
    entry_count = 0
    event_dict = []
    for ifile in file_list:
        eve_ak = ak.from_parquet(ifile)
        event_dict.append(eve_ak)
        entry_count += len(eve_ak)
        total_entry += len(eve_ak)
        if entry_count > chunksize:
            if len(event_dict) > 1:    
                merged_ak = ak.concatenate(event_dict,axis=0)
            elif len(event_dict) == 1:
                merged_ak = event_dict[0]
            else:
                raise Exception("No event in event_dict")
            ak.to_parquet(merged_ak,f"{file_folder}/{outprefix}_{file_count}.parquet")
            # logger.info("File: %s, #entry: %s",file_count,entry_count)
            print("File: %s, #entry: %s" % (file_count,entry_count))
            
            del merged_ak
            file_count += 1
            entry_count = 0
            event_dict.clear()
    
    # be careful, the remaining events should be saved
    if entry_count > 0:
        if len(event_dict) > 1:    
            merged_ak = ak.concatenate(event_dict,axis=0)
        elif len(event_dict) == 1:
            merged_ak = event_dict[0]
        else:
            raise Exception("No event in event_dict")
        ak.to_parquet(merged_ak,f"{file_folder}/{outprefix}_{file_count}.parquet")
        # logger.info("File: %s, #entry: %s",file_count,entry_count)
        print("File: %s, #entry: %s" % (file_count,entry_count))

        del merged_ak
        file_count += 1
        entry_count = 0
        event_dict.clear()

    # delete pre-merge files
    for ifile in file_list:
        os.remove(ifile)
    # logger.info("Total files: %s, total entries: %s",file_count,total_entry)
    tic = time.monotonic()
    print(
        f"""
=======================
 Total files: {file_count}
 Total entries: {total_entry}
 Total time: {np.round(tic-toc,4)} s
=======================
        """
    )
    return

def get_run(fdict, cfg):
    toc = time.monotonic()
    try:
        with open(hcom.abs_path(f"rundoc/{cfg['channel']}_{cfg['year']}_{cfg['version']}_{cfg['step']}_count.yaml"), 'r') as f:
            ncount_cfg = yaml.load(f, Loader=yaml.FullLoader)
    except:
        ncount_cfg = {}

    # output file
    out_path = f"{cfg['out_dir']}/{cfg['version']}/{cfg['step']}/{cfg['year']}/"
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    results = running(fdict, cfg)
    try:
        for isp in results['ntot']:
            new_rlt = {
                'ntot': str(results['ntot'][isp]),
                'npos': str(results['npos'][isp]),
                'nneg': str(results['nneg'][isp]),
                'neff': str(results['neff'][isp]),
                'npass': str(results['npass'][isp]),
            }
            logger.info("<<< END <<< sample: %s, ntot: %s, npos: %s, nneg: %s, neff: %s, npass: %s",isp,new_rlt['ntot'],new_rlt['npos'],new_rlt['nneg'],new_rlt['neff'],new_rlt['npass'])
            ncount_cfg[isp] = new_rlt
        with open(hcom.abs_path(f"rundoc/{cfg['channel']}_{cfg['year']}_{cfg['version']}_{cfg['step']}_count.yaml"), 'w') as f:
            yaml.dump(ncount_cfg, f, default_flow_style=False)
    except:
        logger.warning("Results don't have ntot, npos, nneg, neff, npass")
        pass

    # store the results
    sample_type = "data" if cfg['data'] else "mc"
    with open(f"{out_path}/results_{cfg['channel']}_{cfg['year']}_{cfg['version']}_{cfg['step']}_{sample_type}.pkl", "wb") as f:
        pickle.dump(results, f)

    tic = time.monotonic()
    logger.info(
        f"""
================================
 Total cost time: {np.round(tic-toc,4)} s
================================
        """
    )
    # merge the parquet files
    if cfg['merge'] and cfg['format'] == 'parquet':
        for isp in results['npass']:
            if results['npass'][isp] > 0:
                merge_parquet(f"{out_path}/{isp}")
    return True

