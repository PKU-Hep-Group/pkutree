def get_step_module(fdict, cfg):
    if cfg['channel'] == 'hmm':
        if cfg['step'] == 'obj_sel':
            from module.hmm.sel_module import obj_sel
            return fdict, obj_sel(year=cfg['year'],data=cfg['data'])
        elif cfg['step'] == 'vbf_sel':
            from module.hmm.sel_module import vbf_sel
            fdict_new = {}
            norm_dict = {}
            for isp in fdict.keys():
                fdict_new[isp] = fdict[isp]['files']
                norm_dict[isp] = fdict[isp]['metadata']['norm']
            return fdict_new, vbf_sel(year=cfg['year'],data=cfg['data'],norm_dict=norm_dict)
    return
