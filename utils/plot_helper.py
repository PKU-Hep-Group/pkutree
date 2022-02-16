import os
import os.path as osp
from matplotlib.pyplot import get
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
from hist import Hist
from template.temp_class import super_hist
import mplhep as hep
import matplotlib.pyplot as plt
import matplotlib.ticker as tik
from typing import Optional

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
        'nworker': args.nworker,
        'chunksize': args.chunksize,
        'schema': args.schema,
        'executor': args.executor,
        'config': args.config,
        'nuisance': args.nuisance,
        'sum': args.sum,
        'all': args.all,
        'reorganize': args.reorganize,
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
      'nworker' : %s,
    'chunksize' : %s,
       'schema' : %s,
     'executor' : %s,
       'config' : %s,
     'nuisance' : %s,
          'sum' : %s,
          'all' : %s,
   'reorganize' : %s,
--------------------------------
        """,
        args.mode,
        args.year,
        args.channel,
        args.step,
        args.redo,
        args.format,
        args.nworker,
        args.chunksize,
        args.schema,
        args.executor,
        args.config,
        args.nuisance,
        args.sum,
        args.all,
        args.reorganize,
    )
    return cfg

def get_variable_key(result, exclude_list=['ntot', 'npos', 'nneg', 'neff', 'npass']):
    """
    remove keys in exclude_list, the kept keys should be histogram names
    """
    key_list = list(result.keys())
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

def get_nuisance_cfg(nui_cfg):
    nuisance_content = imp.import_module(nui_cfg)
    nui_dict = nuisance_content.nuisances
    return nui_dict

def get_file_dict(cfg):
    sample_content = imp.import_module(cfg['samples_cfg'])
    sp_dict = sample_content.samples
    if cfg['nuisance']:
        nui_dict = get_nuisance_cfg(cfg['nuisances_cfg'])
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
    if cfg['reorganize']:
        fhist_dict,group_dict = get_plot_cfg(cfg['plot_cfg'])
        output_file_name_reorganize = f"{cfg['out_hist']}/reorganize_results_{cfg['channel']}_{cfg['year']}.pkl"
        reorganize_hist(cfg,results,group_dict,output_file_name_reorganize)
    tic = time.monotonic()
    logger.info(
        """
================================
 Total cost time: %s s
 The results: %s
 The reorganized results: %s
================================
        """,
        np.round(tic - toc,4),
        output_file_name,
        output_file_name_reorganize,
    )
    return True

def get_plot_cfg(plot_cfg):
    plotcfg_content = imp.import_module(plot_cfg)
    fhist_dict = plotcfg_content.results
    group_dict = plotcfg_content.group
    return fhist_dict,group_dict

def merge_hist(hists,subsamples,postfix=""):
    htmp = 0
    for isub in subsamples:
        htmp += hists[f"{isub}{postfix}",:]*subsamples[isub]
    return htmp

def reorganize_hist(cfg,hists,groups,store_path=None):
    hvar_list = get_variable_key(hists)
    year_list = get_year_key(groups)

    # get nuisance
    if cfg['nuisance']:
        nui_dict = get_nuisance_cfg(cfg['nuisances_cfg'])
    year = cfg['year']
    if year not in year_list:
        logger.warning("Year %s not in groups",year)
        raise ValueError
    # hist_dict structure:
    # hist_dict['mass']['DYJetsToLL_M-50']['nominal'] => super_hist
    # hist_dict['mass']['DYJetsToLL_M-50']['CMS_scale_mu']['up'] => hist
    hist_dict = {} # histogram dictionary: hist_dict[hist_name][group_name]
    for ivar in hvar_list:
        all_in_hist = Hist(hists[ivar].to_boost())
        hist_dict[ivar] = {}
        for igp in groups:
            hist_dict[ivar][igp] = {}
            # nominal
            subsample = groups[igp]['subsample'][year]
            group_cfg = copy.deepcopy(groups[igp])
            group_cfg.pop('subsample')
            group_cfg['hist']=merge_hist(all_in_hist,subsample)
            hist_dict[ivar][igp]['nominal'] = super_hist(**group_cfg)
            # shape variations
            for inui in nui_dict:
                hist_dict[ivar][igp][inui] = {}
                if nui_dict[inui]['type']=='shape':
                    hist_dict[ivar][igp][inui]['up'] = merge_hist(all_in_hist,subsample,f"_{inui}_up")
                    hist_dict[ivar][igp][inui]['down'] = merge_hist(all_in_hist,subsample,f"_{inui}_down")
                # FIXME: how to deal with weights?
                # if nui_dict[inui]['type']=='weight':
                #     hist_dict[ivar][igp][inui]['down'] = merge_hist(all_in_hist,subsample,f"_{inui}_var{}")
    if store_path:
        with open(store_path, "wb") as f:
            pickle.dump(hist_dict, f)

    return hist_dict

def plotting(cfg,hist_dict):
    toc = time.monotonic()
    hep.cms.style.CMS["legend.handlelength"]=0.7
    hep.cms.style.CMS["legend.handleheight"]=0.7
    hep.cms.style.CMS["legend.fontsize"]=12
    hep.cms.style.CMS["legend.labelspacing"]=0.4
    hep.cms.style.CMS["font.size"]=15
    hep.cms.style.CMS["axes.labelsize"]=12
    hep.cms.style.CMS["xtick.major.size"]=8
    hep.cms.style.CMS["xtick.minor.size"]=4
    hep.cms.style.CMS["xtick.major.pad"]=4
    hep.cms.style.CMS["xtick.top"]=True
    hep.cms.style.CMS["xtick.major.top"]=True
    hep.cms.style.CMS["xtick.major.bottom"]=True
    hep.cms.style.CMS["xtick.minor.top"]=True
    hep.cms.style.CMS["xtick.minor.bottom"]=True
    hep.cms.style.CMS["xtick.minor.visible"]=True
    hep.cms.style.CMS["ytick.major.size"]=8
    hep.cms.style.CMS["ytick.minor.size"]=4
    hep.cms.style.CMS["ytick.right"]=True
    hep.cms.style.CMS["ytick.major.left"]=True
    hep.cms.style.CMS["ytick.major.right"]=True
    hep.cms.style.CMS["ytick.minor.left"]=True
    hep.cms.style.CMS["ytick.minor.right"]=True
    hep.cms.style.CMS["ytick.minor.visible"]=True
    # hep.cms.style.CMS["axes.linewidth"]=2
    plt.style.use(hep.style.CMS)

    os.makedirs(cfg['out_plot'],exist_ok=True)


    for ivar in hist_dict:
        f, ax = plt.subplots(2,1,figsize=(5,5),gridspec_kw={'height_ratios': [3.3,0.7]},sharex=True)
        hist_list = []
        label_list = []
        color_list = []
        for igp in hist_dict[ivar]:
            if hist_dict[ivar][igp]['nominal'].data:
                continue
            hist_list.append(hist_dict[ivar][igp]['nominal'].hist)
            label_list.append(hist_dict[ivar][igp]['nominal'].label)
            color_list.append(hist_dict[ivar][igp]['nominal'].color)
        zipped = zip(hist_list, label_list, color_list)
        resort_zipped = sorted(zipped, key=lambda x: x[0].sum().value, reverse=False)
        hist_tuple, label_tuple, color_tuple = zip(*resort_zipped)
        hep.histplot(list(hist_tuple), histtype='fill', stack=True, label=list(label_tuple), color=list(color_tuple), ax=ax[0])
        hep.histplot(hist_dict[ivar]['data']['nominal'].hist, histtype='errorbar', stack=False, label=hist_dict[ivar]['data']['nominal'].label, color=hist_dict[ivar]['data']['nominal'].color, ax=ax[0], marker='o', markersize=2,elinewidth=1)
        # ax.legend(leg_handles_new,leg_labels_new,loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
        ax[0].set_xlim([hist_dict[ivar]['data']['nominal'].hist.axes[0].edges[0],hist_dict[ivar]['data']['nominal'].hist.axes[0].edges[-1]])
        x_label = ax[0].get_xlabel()
        ax[0].set_xlabel('', ha='right', x=1.0)
        ax[0].set_ylabel("Events/bin",loc='top')
        # ax[0].set_ylim(top=ax[0].get_ylim()[1]*1000) # minor stick will not be draw if the y-range is very large
        # an additional way to make reasonable y-axis
        ax[0].axhline(y=ax[0].get_ylim()[1]*1000, color='black', linestyle='--', linewidth=1, alpha=0)
        ax[0].set_yscale('log')
        # https://stackoverflow.com/questions/44078409/matplotlib-semi-log-plot-minor-tick-marks-are-gone-when-range-is-large
        locmin = tik.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=64)
        ax[0].yaxis.set_minor_locator(locmin)
        ax[0].yaxis.set_minor_formatter(tik.NullFormatter())
        ax[0].legend(ncol=3,loc='upper left')
        ax[1].set_xlabel(x_label, ha='right', x=1.0)
        # yticks = ax[1].yaxis.get_major_ticks()
        # yticks[0].label1.set_visible(False)
        # yticks[-1].label1.set_visible(False)
        plt.subplots_adjust(hspace=0.03)
        hep.cms.label(label="Preliminary",loc=0,data=True,year='2018',ax=ax[0]) # Preliminary
        plt.savefig(f"{cfg['out_plot']}/LOG_{cfg['channel']}_{cfg['year']}_{ivar}.pdf",bbox_inches='tight')


    for ivar in hist_dict:
        f, ax = plt.subplots(2,1,figsize=(5,5),gridspec_kw={'height_ratios': [3.3,0.7]},sharex=True)
        hist_list = []
        label_list = []
        color_list = []
        for igp in hist_dict[ivar]:
            if hist_dict[ivar][igp]['nominal'].data:
                continue
            hist_list.append(hist_dict[ivar][igp]['nominal'].hist)
            label_list.append(hist_dict[ivar][igp]['nominal'].label)
            color_list.append(hist_dict[ivar][igp]['nominal'].color)
        zipped = zip(hist_list, label_list, color_list)
        resort_zipped = sorted(zipped, key=lambda x: x[0].sum().value, reverse=True)
        hist_tuple, label_tuple, color_tuple = zip(*resort_zipped)
        hep.histplot(list(hist_tuple), histtype='fill', stack=True, label=list(label_tuple), color=list(color_tuple), ax=ax[0])
        hep.histplot(hist_dict[ivar]['data']['nominal'].hist, histtype='errorbar', stack=False, label=hist_dict[ivar]['data']['nominal'].label, color=hist_dict[ivar]['data']['nominal'].color, ax=ax[0], marker='o', markersize=2,elinewidth=1)
        # ax.legend(leg_handles_new,leg_labels_new,loc='upper left', ncol=1, bbox_to_anchor=(1, 1))
        ax[0].set_xlim([hist_dict[ivar]['data']['nominal'].hist.axes[0].edges[0],hist_dict[ivar]['data']['nominal'].hist.axes[0].edges[-1]])
        x_label = ax[0].get_xlabel()
        ax[0].set_xlabel('', ha='right', x=1.0)
        ax[0].set_ylabel("Events/bin",loc='top')
        # set y-axis limits
        ax[0].set_ylim([0.00001,ax[0].get_ylim()[1]*1.4])
        ax[0].legend(ncol=3,loc='upper left')
        ax[1].set_xlabel(x_label, ha='right', x=1.0)
        # yticks = ax[1].yaxis.get_major_ticks()
        # yticks[0].label1.set_visible(False)
        # yticks[-1].label1.set_visible(False)
        plt.subplots_adjust(hspace=0.03)
        hep.cms.label(label="Preliminary",loc=0,data=True,year='2018',ax=ax[0]) # Preliminary
        plt.savefig(f"{cfg['out_plot']}/ORI_{cfg['channel']}_{cfg['year']}_{ivar}.pdf",bbox_inches='tight')

    tic = time.monotonic()
    logger.info(
    """
================================
 Plotting cost time: %s s
 Please find the plots here: %s
================================
    """,
    np.round(tic - toc,4),
    cfg['out_plot'],
    )
    return

def get_plot(cfg):

    fhist_dict,group_dict = get_plot_cfg(cfg['plot_cfg'])
    rlts = {}
    for iyear in fhist_dict:
        with open(fhist_dict[iyear], "rb") as f:
            rlts[iyear] = pickle.load(f)

    hist_dict = {}
    for iyear in rlts:
        if cfg['reorganize']:
            hist_dict[iyear] = reorganize_hist(cfg,rlts[iyear],group_dict,output_file_name_reorganize)
            output_file_name_reorganize = f"{cfg['out_hist']}/reorganize_results_{cfg['channel']}_{cfg['year']}.pkl"
        else:
            hist_dict[iyear] = rlts[iyear]

    for iyear in hist_dict:
        plotting(cfg,hist_dict[iyear])
