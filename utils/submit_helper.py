import json
import os
import os.path
import re
import subprocess
import sys
import random
import yaml
from pathlib import Path
import shutil

def get_cfg(args):
    cfg = {
        'year': args.year,
        'channel': args.channel,
        'nano_ver': args.version,
        'sample_type': args.type,
        'dryrun': args.dryrun,
        'all': args.all,
        'sample': args.sample,
        'jflavour': args.flavour,
        'pre': args.pre,
        'step': args.step,
        'redo': args.redo,
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
    job_dir = condor_cfg['job_dir']
    in_dir = condor_cfg['in_dir']  # input area
    out_dir = condor_cfg['out_dir']  # output ntuple area

    with open('./config/step_cfg.yaml', 'r') as f:
        step_cfg = yaml.load(f, Loader=yaml.FullLoader)
        step_handle = step_cfg[cfg['channel']][cfg['step']]
    with open(f"./datasets/{ds_cfg[cfg['channel']][cfg['nano_ver']][cfg['sample_type']][int(cfg['year'])]}", 'r') as f:
        ds_yml = yaml.load(f, Loader=yaml.FullLoader)
    
    cfg['job_dir'] = job_dir
    cfg['in_dir'] = in_dir
    cfg['out_dir'] = out_dir
    cfg['ds_yml'] = ds_yml
    cfg['step_handle'] = step_handle
    return cfg


def get_file_info(fpath):
    file_dict = {}
    for path in Path(fpath).rglob('*.root'):
        # print(path.resolve())
        tmp_path = str(path.resolve())
        file_dict[tmp_path.split("/")[-1]] = tmp_path
    return file_dict


def get_sub(job_path, sample, job_flavour="testmatch",idx=0,long_queue=None):
    if long_queue is None:
        long_queue = []
    # print("===> get_sub", "job_path", job_path, "sample:", sample, "idx:", idx, "long_queue:", len(long_queue) != 0)
    if len(long_queue) == 0:
        job_name = f"{sample}_{idx}"
        file_content = f"executable = {job_path}/{job_name}.sh\n"
        file_content += "universe = vanilla\n"
        file_content += f"output = {job_path}/{job_name}.out\n"
        file_content += f"error = {job_path}/{job_name}.err\n"
        file_content += f"log = {job_path}/{job_name}.log\n"
        # file_content += "request_cpus = 1\n"
        # file_content += "request_memory = 1024\n"
        # file_content += "request_disk = 1024\n"
        # file_content += "requirements = (machine == \"atlas.phy.pku.edu.cn\") || (machine == \"farm.phy.pku.edu.cn\") || (machine == \"node01.phy.pku.edu.cn\") || (machine == \"node02.phy.pku.edu.cn\") || (machine == \"node03.phy.pku.edu.cn\") || (machine == \"node04.phy.pku.edu.cn\") || (machine == \"node05.phy.pku.edu.cn\") || (machine == \"node06.phy.pku.edu.cn\")\n"
        file_content += f"+JobFlavour = \"{job_flavour}\"\n"
        file_content += "queue\n"
        tmp = open(f"{job_path}/{job_name}.sub", "w")
        tmp.write(file_content)
    else:
        file_content = f"executable = {job_path}/$(JNAME).sh\n"
        file_content += "arguments = $(JNAME) $(INFILE)\n"
        file_content += "universe = vanilla\n"
        file_content += f"output = {job_path}/$(JNAME).out\n"
        file_content += f"error = {job_path}/$(JNAME).err\n"
        file_content += f"log = {job_path}/$(JNAME).log\n"
        # file_content += "request_cpus = 1\n"
        # file_content += "request_memory = 1024\n"
        # file_content += "request_disk = 1024\n"
        # file_content += "requirements = (machine == \"atlas.phy.pku.edu.cn\") || (machine == \"farm.phy.pku.edu.cn\") || (machine == \"node01.phy.pku.edu.cn\") || (machine == \"node03.phy.pku.edu.cn\")\n"
        file_content += f"+JobFlavour = \"{job_flavour}\"\n"
        file_content += "queue JNAME in (\n"
        for i in long_queue:
            file_content += f"{i}\n"
        file_content += ")\n"

        # print(file_content)
        proc = subprocess.Popen(["condor_submit"], shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8')
        out, err = proc.communicate(input=file_content)
        if proc.returncode != 0:
            sys.stderr.write(err)
            raise RuntimeError('Job submission failed.')
        print(out.strip())

        matches = re.match('.*submitted to cluster ([0-9]*).', out.split('\n')[-2])
        if not matches:
            sys.stderr.write('Failed to retrieve the job id. Job submission may have failed.\n')
            for i in long_queue:
                jidFile = f"{job_path}/{i}.jid"
                open(jidFile, 'w').close()
        else:
            clusterId = matches.group(1)
            # now write the jid files
            proc = subprocess.Popen(['condor_q', clusterId, '-l', '-attr', 'ProcId,Cmd', '-json'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
            out, err = proc.communicate(input=None)
            try:
                qlist = json.loads(out.strip())
            except:
                sys.stderr.write('Failed to retrieve job info. Job submission may have failed.\n')
                for jName in long_queue:
                    jidFile = f"{job_path}/{jName}.jid"
                    open(jidFile, 'w').close()
            else:
                for qdict in qlist:
                    with open(qdict['Cmd'].replace('.sh', '.jid'), 'w') as out:
                        out.write('%s.%d\n' % (clusterId, qdict['ProcId']))
    # print("<=== get_sub END :)")
    return


def get_sh(idx, file, sample, job_path, out_path, cfg):
    HOME_PATH = os.environ['HOME']
    # print("===> get_sh")
    file_content = "#!/bin/bash\n"
    is_data = "-d" if cfg['sample_type'] == "data" else ""
    file_content += """
    """
    file_content += f"""
echo ">>> conda initialize >>>"
source {HOME_PATH}/miniconda3/bin/activate
conda activate xcu
echo "<<< conda initialize <<<"

echo ">>> analysing >>>"
python -m {cfg['step_handle']} -i {file} -o {out_path} -y {cfg['year']} -s {cfg['step']} -sp {sample} {is_data}
echo "<<< analysing <<<"
    \n"""

    file_content += f"[ $? -eq 0 ] && mv {job_path}/{sample}_{idx}.jid {job_path}/{sample}_{idx}.done\n"
    tmp = open(f"{job_path}/{sample}_{idx}.sh", "w")
    # print(f"{job_path}/{sample}_{idx}.sh")
    tmp.write(file_content)
    os.system(f"chmod +x {job_path}/{sample}_{idx}.sh")
    # print("<=== get_sh END :)")
    return


def submit_command(cfg, sample=None):
    print("===> submit_command", "year:", cfg['year'], "sample:", sample)

    job_path = f"{cfg['job_dir']}/{cfg['channel']}_{cfg['year']}_{cfg['nano_ver']}_{cfg['step']}/{sample}/"
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
    if not os.path.exists(job_path):
        os.makedirs(job_path)

    file_dict = get_file_info(in_path)
    print("---", "sample:", sample, ", #file:", len(file_dict))
        
    _long_queue = []
    for idx,ifname in enumerate(file_dict):
        # We write the SUB file for documentation / resubmission, but initial submission will be done in one go below
        get_sub(job_path, sample, cfg['jflavour'],idx)
        get_sh(idx, file_dict[ifname], sample, job_path, out_path, cfg)
        _long_queue.append(f"{sample}_{idx}")
    if cfg['dryrun']:
        pass
    else:
        get_sub(job_path, sample, cfg['jflavour'], long_queue=_long_queue)  # dryrun will just generate submit files, but not run
    print("<=== submit_command END :)")
    return True


def status_command(cfg, sample):
    print("===> status_command", "year:", cfg['year'], "sample:", sample)
    use_gfal, file_dict = get_file_info(cfg['ds_yml'][sample]['dataset'])
    print("---", "sample:", sample, ", use_fal:", use_gfal, ", #file:", len(file_dict))

    # print(file_dict)
    sname = cfg['ds_yml'][sample]['dataset']
    if sname.startswith("gsiftp://") or sname.startswith("local:"):
        if sname.endswith("/"):
            sname = sname.split("/")[-2]
        else:
            sname = sname.split("/")[-1]
        out_path = f"{cfg['out_dir']}/{cfg['nano_ver']}/private/{sname}/"
    else:
        out_path = f"{cfg['out_dir']}/{cfg['nano_ver']}/{sname}/"

    done_file = []
    not_done_file = []
    for idx,ifname in enumerate(file_dict):
        out_file_name = f"{out_path}/{ifname}"
        if os.path.exists(out_file_name):
            if cfg['checkfile']:
                try:
                    proc = subprocess.Popen([f"gfal-sum {out_file_name} ADLER32"], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, encoding='utf-8')
                    out, err = proc.communicate(input=None)
                    local_sum = out.strip("\n").split(" ")[-1]
                    if file_dict[ifname]['checksum'] == local_sum:
                        print("--- The job done already, check:", out_file_name)
                        done_file.append(out_file_name)
                    else:
                        print(f"xxx Remove bad file: {out_file_name}")
                        os.remove(out_file_name)
                except:
                    print(f"xxx [Error]: could not run gfal-sum, please check, e.g., unset cmsenv")
                    exit(1)
            else:
                print("--- The job done already, check:", out_file_name)
                done_file.append(out_file_name)
                pass
        else:
            not_done_file.append(out_file_name)

    print("--- sample:", sample, ", total files:", len(file_dict), ", done:", len(done_file), "not done:", len(not_done_file), ", they are following jobs:")
    for i in not_done_file:
        print("\t-", i)
    print("<=== submit_command END :)")
    return True
