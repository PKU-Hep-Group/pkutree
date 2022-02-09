import argparse
import utils.common_helper as hcom
import utils.run_helper as hrun
import logging


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-y', '--year', help='which year to run？Default: 2018', choices=('2016','2017','2018'),  default='2018')
parser.add_argument('-c', '--channel', help='which channel: vhjj, ssww, your own channel, ...', default='vhjj')
parser.add_argument('-d', '--data', help='is data or mc, default: mc', action='store_true', default=False)
parser.add_argument('-v', '--version', help='which nanoAOD version: nanov7, nanov8, nanov9, ..., default: nanov9', default='nanov9')
parser.add_argument('-r', '--redo', help='redo the job, default: False', action='store_true', default=False)
parser.add_argument('-s', '--sample', help='run on which sample, can be more than 1', nargs='+', default=None)
parser.add_argument('-a', '--all', help='run on all samples, default: False', action='store_true', default=False)
parser.add_argument('-f', '--format', help='input file format, default is root', choices=('root','parquet'), default='root')
parser.add_argument('-n', '--norm', help="get normlization of each sample, the doc of 'obj_sel' step needed", action='store_true', default=False)
parser.add_argument('--pre', help='the name of the pre-step', default=None)
parser.add_argument('--step', help='which step to run', default='obj_sel')
parser.add_argument('--nworker', help='number of workers, default is 100', default='100')
parser.add_argument('--chunksize', help='chunksize, default is 300000', default='300000')
parser.add_argument('--schema', help='which schema? Default: NanoAODSchema, others: BaseSchema', choices=('NanoAODSchema','BaseSchema'), default='NanoAODSchema')
parser.add_argument('--executor', help='which executor? Default: FuturesExecutor, others: IterativeExecutor, DaskExecutor', choices=('FuturesExecutor','IterativeExecutor','DaskExecutor'), default='FuturesExecutor')
# ,default=['WpWpJJ_EWK','WpWpJJ_EWK_powheg','WmWmJJ_EWK_powheg'])
args = parser.parse_args()

# logging settings
hcom.setup_logging(hcom.abs_path("config/logging_cfg.yaml"))
logger = logging.getLogger('main')

def to_run(cfg):
    logger.info(">>> start running >>>")

    # file dict
    fdict = hrun.get_file_dict(cfg)
    if len(fdict) < 1:
        logger.warning("No samples to run")
        return False
    hrun.get_run(fdict, cfg)
    logger.info("<<< end running <<<")
    return True

if __name__ == '__main__':
    if (not args.sample) and (not args.all):
        logger.warning("[Warning] No sample to run, please specify the sample(s), exit now")
        exit(1)
    cfg = hrun.get_cfg(args)
    # exit(1)
    to_run(cfg)

