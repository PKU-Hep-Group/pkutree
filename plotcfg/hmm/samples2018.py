# example of samples file

import utils.plot_helper as hplot

# common settings start
year = '2018'
base_path = '/data/pubfs/xiaoj/hmm/2ntuple/nanov7/obj_sel/2018'
# common settings end

# the sample dict
samples = {}

# data
data_list = [
    'SingleMuon_Run2018A',
    'SingleMuon_Run2018B',
    'SingleMuon_Run2018C',
    'SingleMuon_Run2018D',
]
samples = hplot.get_sample_info(samples,None,slist=data_list,spath=base_path,data=True,year=year)
# signal samples
signal_list = [
    # VBF H
    'VBFHToMuMu_M-120',
    'VBFHToMuMu_M-125',
    'VBFHToMuMu_M-125_withDipoleRecoil',
    'VBFHToMuMu_M-130',
    # ggH
    'GluGluHToMuMu_M125',
    'GluGluHToMuMu_M125_TuneCP5down',
    'GluGluHToMuMu_M125_TuneCP5up',
    'GluGluToContinToZZTo2e2mu',
    'GluGluToContinToZZTo2e2tau',
    'GluGluToContinToZZTo2mu2nu',
    'GluGluToContinToZZTo2mu2tau',
    'GluGluToContinToZZTo4mu',
    'GluGluToContinToZZTo4tau',    
]
samples = hplot.get_sample_info(samples,None,slist=signal_list,spath=base_path,signal=True,year=year,merge_ext=True)

# background samples
background_list = [
    # DY
    'DYJetsToLL_M-105To160',
    'DYJetsToLL_M-105To160_VBFFilter',
    'DYJetsToLL_M-105To160_VBFFilter_ext1',
    'DYJetsToLL_M-50',
    'DYJetsToLL_M-50_ext1',
    # VBF Z
    'EWK_LLJJ_MLL-50_MJJ-120',
    'EWK_LLJJ_MLL_105-160',
    'EWK_LLJJ_MLL_105-160_MG',
    # top
    'ST_s-channel_antitop',
    'ST_s-channel_top',
    'ST_t-channel_antitop',
    'ST_t-channel_top',
    'ST_tW_antitop',
    'ST_tW_top',
    'TTTJ',
    'TTTT',
    'TTTT_ext1',
    'TTTW',
    'TTTo2L2Nu',
    'TTToSemiLeptonic',
    'TTToSemiLeptonic_ext1',
    'TTWJetsToLNu',
    'TTWW',
    'TTWW_ext1',
    'TTZToLLNuNu_M-10',
    'tZq_ll',
    # VV
    'WWTo2L2Nu',
    'WZTo2L2Q',
    'WZTo3LNu',
    'WZTo3LNu_MG',
    'WZTo3LNu_MG_ext1',
    'ZZTo2L2Nu',
    'ZZTo2L2Q',
    'ZZTo4L',
    'ZZTo4L_ext1',
    # VVV
    'WWW',
    'WWZ',
    'WZZ',
    'ZZZ',
]
samples = hplot.get_sample_info(samples,None,slist=background_list,spath=base_path,data=False,year=year,merge_ext=True)
