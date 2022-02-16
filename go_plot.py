import argparse
import utils.common_helper as hcom
import utils.run_helper as hrun
import utils.plot_helper as hplot
import logging


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('mode', help='which mode? Default: hist', choices=('hist','plot'),  default='plot')
parser.add_argument('-y', '--year', help='which year to run? Default: 2018', choices=('2016','2017','2018'),  default='2018')
parser.add_argument('-c', '--channel', help='which channel: vhjj, ssww, your own channel, ...', default='vhjj')
parser.add_argument('-s', '--step', help='which step, will decide which module to use', default='vbf_sel')
parser.add_argument('-r', '--redo', help='redo the job, default: False', action='store_true', default=False)
parser.add_argument('-f', '--format', help='input file format, default is root', choices=('root','parquet'), default='root')
parser.add_argument('--nworker', help='number of workers, default is 100', default='100')
parser.add_argument('--chunksize', help='chunksize, default is 300000', default='300000')
parser.add_argument('--schema', help='which schema? Default: NanoAODSchema, others: BaseSchema', choices=('NanoAODSchema','BaseSchema'), default='NanoAODSchema')
parser.add_argument('--executor', help='which executor? Default: FuturesExecutor, others: IterativeExecutor, DaskExecutor', choices=('FuturesExecutor','IterativeExecutor','DaskExecutor'), default='FuturesExecutor')
parser.add_argument('--config', help='config name', default='config.py')
parser.add_argument('--nuisance', help='consider nuisances, default: False', action='store_true', default=False)
parser.add_argument('--sum', help='sum plots from all years', action='store_true', default=False)
parser.add_argument('--all', help='plot all years', action='store_true', default=False)
parser.add_argument('--reorganize', help='re-organize the plots for plotting', action='store_true', default=False)
# ,default=['WpWpJJ_EWK','WpWpJJ_EWK_powheg','WmWmJJ_EWK_powheg'])
args = parser.parse_args()

# logging settings
hcom.setup_logging(hcom.abs_path("config/logging_cfg.yaml"))
logger = logging.getLogger('plot')

def to_hist(cfg):
    logger.info(">>> start histograming >>>")

    # file dict
    fdict = hplot.get_file_dict(cfg)
    # for i in fdict:
    #     print(i, fdict[i])
    # exit(0)
    if len(fdict) < 1:
        logger.warning("No samples to run")
        return False
    hplot.get_hist(fdict, cfg)
    logger.info("<<< end histograming <<<")
    return True

def to_plot(cfg):
    logger.info(">>> start plotting >>>")
    hplot.get_plot(cfg)
    logger.info("<<< end plotting <<<")
    return True

if __name__ == '__main__':
    cfg = hplot.get_cfg(args)
    if args.mode == 'hist':
        to_hist(cfg)
    elif args.mode == 'plot':
        to_plot(cfg)
    else:
        logger.error("[Error] Invalid mode '%s', terminate now", args.mode)
