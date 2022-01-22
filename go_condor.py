import argparse
import yaml
import utils.submit_helper as sh

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('mode', help='which mode: look, download, status', choices=('look', 'skim', 'status'), default='status')
parser.add_argument('-t','--type', help='which type: data, mc', choices=('data', 'mc'), default='mc')
parser.add_argument('-v','--version', help='which nanoAOD version: nanov7, nanov8, nanov9, ..., default: nanov9', default='nanov9')
parser.add_argument('-c', '--channel', help='which channel: vhjj, ssww, your own channel, ...', default='vhjj')
parser.add_argument('-y', '--year', help='which year to run: 2016, 2017, 2018', default='2018')
parser.add_argument('-f', '--flavour', help='job flavour: espresso, longlunch, workday, ...', default='testmatch')
parser.add_argument('-r', '--redo', help='redo the job, default: False', action='store_true', default=False)
parser.add_argument('--pre', help='the name of the pre-step', default=None)
parser.add_argument('--step', help='which step to run', default='obj_sel')
parser.add_argument('--dryrun', help='only generate submit files, but not run', action='store_true', default=False)
# avoid downloading at the same time
group = parser.add_mutually_exclusive_group()  # type: argparse._MutuallyExclusiveGroup
group.add_argument('-a', '--all', help='run on all, default: True', action='store_false', default=True)
group.add_argument('-s', '--sample', help='run on which sample, can be more than 1', nargs='+')
# ,default=['WpWpJJ_EWK','WpWpJJ_EWK_powheg','WmWmJJ_EWK_powheg'])
args = parser.parse_args()



def to_skim(year, sample_type):
    print("===> to_submit:", "year:", year, "sample_type:", sample_type)
    cfg = sh.get_cfg(args)

    if cfg['sample']:
        for isp in cfg['sample']:
            if isp in cfg['ds_yml']:
                sh.submit_command(cfg, isp)
            else:
                print("--- submit info:", "no sample: ", isp, "in", year)
    else:
        for isp in cfg['ds_yml']:
            sh.submit_command(cfg, isp)
    print("<=== to_submit END :)")


def to_status(year, sample_type):
    print("===> to_status: year:", year, "sample_type:", sample_type)
    cfg = sh.get_cfg(args)
    if cfg['sample']:
        for isp in cfg['sample']:
            if isp in cfg['ds_yml']:
                sh.status_command(cfg, isp)
            else:
                print("--- status info:", "no sample: ", isp, "in", year)
    else:
        for isp in cfg['ds_yml']:
            sh.status_command(cfg, isp)
    print("<=== to_status END :)")


def have_a_look(ds):
    use_gfal, file_dict = sh.get_file_info(ds)
    for ifile, ifname in enumerate(file_dict):
        print("===> File index:", ifile, ", file name:", ifname)
        print("<=== Available path:",file_dict[ifname])


if __name__ == '__main__':
    if args.mode == "look":
        have_a_look(args.dataset)
    elif args.mode == 'skim':
        to_skim(args.year, args.type)
    elif args.mode == 'status':
        to_status(args.year, args.type)
    else:
        print("===>", "no such mode")