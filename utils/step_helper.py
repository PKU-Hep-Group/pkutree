def get_step_module(channel, step):
    if channel == 'hmm':
        if step == 'obj_sel':
            from module.hmm.sel_module import obj_sel as the_processor
            return the_processor
    return
