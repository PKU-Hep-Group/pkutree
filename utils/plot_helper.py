import os
import os.path as osp
import yaml
from pathlib import Path
import shutil
from coffea.nanoevents import NanoAODSchema, BaseSchema
from coffea import processor
import utils.common_helper as hcom
import utils.step_helper as hstep
import time
import pickle
import importlib as imp
import utils.run_helper as hrun
import copy
import numpy as np

import logging
logger = logging.getLogger('plot_helper')

def get_cfg(args):
    cfg = {
        'mode': args.mode,
        'year': args.year,
        'channel': args.channel,
        'step': args.step,
        'redo': args.redo,
        'format': args.format,
        'input': args.input,
        'nworker': args.nworker,
        'chunksize': args.chunksize,
        'schema': args.schema,
        'executor': args.executor,
        'config': args.config,
        'nuisance': args.nuisance,
    }
    try:
        config_content = imp.import_module(f"plotcfg.{cfg['channel']}.{cfg['config'][:-3]}")
    except:
        logger.error("[Error] Invalid config '%s', terminate now", f"plotcfg.{cfg['channel']}.{cfg['config'][:-3]}")
        exit(1)
    cfg['samples_cfg'] = f"plotcfg.{cfg['channel']}.{config_content.samples_cfg[cfg['year']][:-3]}"
    cfg['nuisances_cfg'] = f"plotcfg.{cfg['channel']}.{config_content.nuisances_cfg[cfg['year']][:-3]}"
    cfg['plot_cfg'] = f"plotcfg.{cfg['channel']}.{config_content.plot_cfg[:-3]}"

    with open(hcom.abs_path('config/path_cfg.yaml'), 'r') as f:
        condor_cfg = yaml.load(f, Loader=yaml.FullLoader)
    plot_base = condor_cfg['plot_dir']    
    cfg['out_hist'] = osp.join(plot_base,config_content.out_hist)
    cfg['out_plot'] = osp.join(plot_base,config_content.out_plot)

    logger.info(
        """
--------------------------------
>>> Following is the options <<<
         'mode' : %s,
         'year' : %s,
      'channel' : %s,
         'step' : %s,
         'redo' : %s,
       'format' : %s,
        'input' : %s,
      'nworker' : %s,
    'chunksize' : %s,
       'schema' : %s,
     'executor' : %s,
       'config' : %s,
     'nuisance' : %s,
--------------------------------
        """,
        args.mode,
        args.year,
        args.channel,
        args.step,
        args.redo,
        args.format,
        args.input,
        args.nworker,
        args.chunksize,
        args.schema,
        args.executor,
        args.config,
        args.nuisance,
    )
    return cfg

def get_hist_key(result, exclude_list=['ntot', 'npos', 'nneg', 'neff', 'npass']):
    """
    remove keys in exclude_list, the kept keys should be histogram names
    """
    key_list = []
    for iyear in result:
        key_list = list(result[iyear].keys())
        break
    for ikey in exclude_list:
        key_list.remove(ikey)
    return key_list

def get_year_key(group):
    """
    get the exist year
    """
    key_list = []
    for ikey in group:
        key_list += list(group[ikey]['subsample'].keys())
        break
    # check if the years in key_list are in all the group
    # if not, remove the year
    for ikey in group:
        for iyear in key_list:
            if iyear not in group[ikey]['subsample'].keys():
                key_list.remove(iyear)
    return key_list

def get_gp_key(group, sample_type):
    """
    sample_type => 0: data, 1: signal, 2: background
    """
    key_list = []
    if sample_type == 0:
        for ikey in group:
            if 'data' in group[ikey]:
                if group[ikey]['data']:
                    key_list.append(ikey)
    elif sample_type == 1:
        for ikey in group:
            if 'signal' in group[ikey]:
                key_list.append(ikey)
    elif sample_type == 2:
        for ikey in group:
            if not 'data' in group[ikey] and not 'signal' in group[ikey]:
                key_list.append(ikey)
    else:
        raise ValueError("sample_type must be 0, 1, or 2")
    
    return key_list

########
# open the yaml file firstly
xsec_yml = {
    '2018': hcom.abs_path(f"datasets/mc_2018_hmm_nanov7.yaml"),
}
count_yml = {
    '2018': hcom.abs_path(f"rundoc/hmm_2018_nanov7_obj_sel_count.yaml"),
}
xsec_dict = {}
for i in xsec_yml:
    with open(xsec_yml[i], 'r') as f:
        xsec_dict[i] = yaml.load(f, Loader=yaml.FullLoader)
count_dict = {}
for i in count_yml:
    with open(count_yml[i], 'r') as f:
        count_dict[i] = yaml.load(f, Loader=yaml.FullLoader)
with open(hcom.abs_path("config/lumi_cfg.yaml"), 'r') as f:
    lumi_cfg = yaml.load(f, Loader=yaml.FullLoader)
 
def get_norm(data,year,sample_list):
    if data:
        return 1.0
    if len(sample_list) == 0:
        return 1.0
    sample_name = sample_list[0]
    sample_xs = xsec_dict[year][sample_name]['xsec']
    sample_neff = 0
    for isp in sample_list:
        sample_neff += float(count_dict[year][isp]['neff'])
    return sample_xs * 1000 * lumi_cfg[year] / sample_neff

def get_sample_info(sdict,smap,slist=None,spath="./",data=False,signal=False,datadriven=False,year='2018',merge_ext=True):
    if slist:
        smap={i:[i] for i in slist}
    if not smap: return None

    sample_map = copy.deepcopy(smap)
    if merge_ext:
        for ikey in smap:
            if "_ext" in ikey:
                sample_map[ikey[:-5]] += sample_map[ikey]
                del sample_map[ikey]
    
    for s in sample_map:
        tmp_files = []
        for isp in sample_map[s]:
            tmp_files += hrun.get_file_list(osp.join(spath,isp),f"*.parquet")
        sdict[s] = {
            "treename": "Events",
            "files": tmp_files,
            "metadata":{
                'data': data,
                'signal': signal,
                'datadriven': datadriven,
                'norm': get_norm(data,year,sample_map[s]),
                'year': year,
            }
        }
    return sdict

def get_branch_map(br_map_dict, up=True):
    new_map = {
        ibr: br_map_dict[ibr]+"_up" if up else br_map_dict[ibr]+"_down" for ibr in br_map_dict
    }
    return new_map

def add_nuisance_dict(sp_dict,nui_dict):
    # for weight type
    for inui in nui_dict:
        if nui_dict[inui]['type'] == 'weight':
            for isp in nui_dict[inui]['samples']:
                sp_info_orig = copy.deepcopy(sp_dict[isp])
                for idx,iwgt in enumerate(nui_dict[inui]['samples'][isp]):
                    sp_info_orig['metadata']['weights']=iwgt
                    sp_dict[f"{isp}_{nui_dict[inui]['name']}_var{idx}"] = copy.deepcopy(sp_info_orig)
        elif nui_dict[inui]['type'] == 'shape':
            for isp in nui_dict[inui]['samples']:
                # up
                sp_info_orig = copy.deepcopy(sp_dict[isp])
                sp_info_orig['metadata']['weights']=nui_dict[inui]['samples'][isp][0]
                sp_info_orig['metadata']['branchmap']=get_branch_map(nui_dict[inui]['map'],True)
                sp_dict[f"{isp}_{nui_dict[inui]['name']}_up"] = copy.deepcopy(sp_info_orig)
                # down
                sp_info_orig = copy.deepcopy(sp_dict[isp])
                sp_info_orig['metadata']['weights']=nui_dict[inui]['samples'][isp][1]
                sp_info_orig['metadata']['branchmap']=get_branch_map(nui_dict[inui]['map'],False)
                sp_dict[f"{isp}_{nui_dict[inui]['name']}_down"] = copy.deepcopy(sp_info_orig)
        else:
            pass
    return sp_dict

def get_file_dict(cfg):
    sample_content = imp.import_module(cfg['samples_cfg'])
    sp_dict = sample_content.samples
    if cfg['nuisance']:
        nuisance_content = imp.import_module(cfg['nuisances_cfg'])
        nui_dict = nuisance_content.nuisances
        final_dict = add_nuisance_dict(sp_dict,nui_dict)
    else:
        final_dict = sp_dict
    return final_dict

def get_hist(fdict, cfg):
    toc = time.monotonic()

    # output file
    if not os.path.exists(cfg['out_hist']):
        os.makedirs(cfg['out_hist'])

    results = hrun.running(fdict, cfg)
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
    except:
        logger.warning("Results don't have ntot, npos, nneg, neff, npass")
        pass

    # store the results
    output_file_name = f"{cfg['out_hist']}/results_{cfg['channel']}_{cfg['year']}.pkl"
    with open(output_file_name, "wb") as f:
        pickle.dump(results, f)

    tic = time.monotonic()
    logger.info(
        """
================================
 Total cost time: %s s
 The results: %s
================================
        """,
        np.round(tic - toc,4),
        output_file_name,
    )
    return True

def get_plot(cfg):
    toc = time.monotonic()

    rlt_hists = {}
    rlt_hists[cfg['year']] = get_hist(get_file_dict(cfg),cfg)
    # output file
