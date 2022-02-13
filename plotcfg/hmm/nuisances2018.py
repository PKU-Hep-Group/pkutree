# example of nuisances file

import utils.plot_helper as hplot
import yaml
import utils.common_helper as hcom
import plotcfg.hmm.samples2018 as samples
import numpy as np

# common settings
year = '2018'
# samples
samples = samples.samples
mc = [isp for isp in samples.keys() if not samples[isp]['metadata']['data'] and not samples[isp]['metadata']['datadriven']]
signal = [isp for isp in samples.keys() if samples[isp]['metadata']['signal']]
data = [isp for isp in samples.keys() if samples[isp]['metadata']['data']]
datadriven = [isp for isp in samples.keys() if samples[isp]['metadata']['datadriven']]
# branch map
with open(hcom.abs_path("plotcfg/hmm/branch_map.yml"), 'r') as f:
    branch_map = yaml.load(f, Loader=yaml.FullLoader)

# nuisances
nuisances = {}
### Luminosity
nuisances['lumi_Uncorrelated'] = {
    'name': 'lumi_13TeV_2018',
    'type': 'lnN',
    'samples': dict((skey, '1.015') for skey in mc),
}
nuisances['lumi_13TeV_Run2'] = {
    'name': 'lumi_13TeV_Run2',
    'type': 'lnN',
    'samples': dict((skey, '1.02') for skey in mc),
}
nuisances['lumi_13TeV_1718'] = {
    'name': 'lumi_13TeV_1718',
    'type': 'lnN',
    'samples': dict((skey, '1.002') for skey in mc),
}

### Muon
# br_map = {
#     'Muon_pt': 'Muon_pt_roccor',
#     'MET_pt': 'MET_pt_roccor',
#     'MET_phi': 'MET_phi_roccor',
# }
nuisances['scale_mu'] = {
    'name': 'CMS_scale_mu',
    'type': 'shape',
    'map': branch_map[year]['Muon_pt_scale'],
    'samples': dict((skey, ['1','1']) for skey in mc),
}

### Jet
jes_dict =[
    'JES_Absolute',
    'JES_Absolute_2018',
    'JES_BBEC1',
    'JES_BBEC1_2018',
    'JES_EC2',
    'JES_EC2_2018',
    'JES_FlavorQCD',
    'JES_HF',
    'JES_HF_2018',
    'JES_RelativeBal',
    'JES_RelativeSample_2018',
    'JES_Total',
]
for i in jes_dict:
    nuisances[i] = {
        'name': 'CMS_'+i,
        'type': 'shape',
        'map': branch_map[year][i],
        'samples': dict((skey, ['1','1']) for skey in mc),
    }

nuisances['JER_2018'] = {
    'name': 'CMS_JER_2018',
    'type': 'shape',
    'map': branch_map[year]['JER'],
    'samples': dict((skey, ['1','1']) for skey in mc),
}

# MET
nuisances['MET_uncluster'] = {
    'name': 'CMS_MET_pt_Uncluster',
    'type': 'shape',
    'map': branch_map[year]['MET_pt_UnclusteredEnergy'],
    'samples': dict((skey, ['1','1']) for skey in mc),
}

# Pileup
nuisances['PU'] = {
    'name': 'CMS_PU_2018',
    'type': 'shape',
    'map': {'PUWeight_nominal':'PUWeight'},
    'samples': dict((skey, ['1','1']) for skey in mc),
}

# Theory uncertainties
# ggF hmm
ggF_hmm = ['GluGluHToMuMu_M125','GluGluHToMuMu_M125_TuneCP5down','GluGluHToMuMu_M125_TuneCP5up']
nuisances['PDF_ggF_hmm'] = {
    'name': 'CMS_PDF_ggF_hmm',
    'type': 'weight',
    'samples': dict((skey, [f'LHEPdfWeight[:,{i}]' for i in range(0,33)]) for skey in ggF_hmm),
}

nuisances['scale_ggF_hmm'] = {
    'name': 'CMS_scale_ggF_hmm',
    'type': 'weight',
    'samples': dict((skey, [f'LHEScaleWeight[:,{i}]' for i in range(0,9)]) for skey in ggF_hmm),
}

nuisances['PS_ggF_hmm'] = {
    'name': 'CMS_PS_ggF_hmm',
    'type': 'weight',
    'samples': dict((skey, [f'PSWeight[:,{i}]' for i in range(0,4)]) for skey in ggF_hmm),
}
# VBF hmm
VBF_hmm = ['VBFHToMuMu_M-125_withDipoleRecoil','VBFHToMuMu_M-125','VBFHToMuMu_M-120','VBFHToMuMu_M-130']
nuisances['PDF_VBF_hmm'] = {
    'name': 'CMS_PDF_VBF_hmm',
    'type': 'weight',
    'samples': dict((skey, [f'LHEPdfWeight[:,{i}]' for i in range(0,33)]) for skey in VBF_hmm),
}

nuisances['scale_VBF_hmm'] = {
    'name': 'CMS_scale_VBF_hmm',
    'type': 'weight',
    'samples': dict((skey, [f'LHEScaleWeight[:,{i}]' for i in range(0,9)]) for skey in VBF_hmm),
}

nuisances['PS_VBF_hmm'] = {
    'name': 'CMS_PS_VBF_hmm',
    'type': 'weight',
    'samples': dict((skey, [f'PSWeight[:,{i}]' for i in range(0,4)]) for skey in VBF_hmm),
}

