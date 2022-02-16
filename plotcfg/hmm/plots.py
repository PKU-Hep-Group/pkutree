# example of plots file
# results include histograms
results = {
    '2018': '/data/pubfs/xiaoj/hmm/plot/hist_hmm_220211/reorganize_results_hmm_2018.pkl',
}
# groups
group = {}
group['data'] = {
    'label': 'Data',
    'color': 'black',
    'scale': 1.0,
    'subsample': {
        '2018': {
            'SingleMuon_Run2018A': 1.0,
            'SingleMuon_Run2018B': 1.0,
            'SingleMuon_Run2018C': 1.0,
            'SingleMuon_Run2018D': 1.0,
        },
    },
    'data': True,
}

group['DY'] = {
    'label': 'DY',
    'color': 'yellow',
    'scale': 1.0,
    'subsample': {
        '2018': {
            'DYJetsToLL_M-50': 1.0,
        },
    },
}

group['VBF_Z'] = {
    'label': 'VBF Z',
    'color': 'purple',
    'scale': 1.0,
    'subsample': {
        '2018': {
            'EWK_LLJJ_MLL-50_MJJ-120': 1.0,
            'EWK_LLJJ_MLL_105-160': 1.0,
        },
    },
}

group['top'] = {
    'label': 'Top',
    'color': 'cyan',
    'scale': 1.0,
    'subsample': {
        '2018': {
            'ST_s-channel_antitop': 1.0,
            'ST_s-channel_top': 1.0,
            'ST_t-channel_antitop': 1.0,
            'ST_t-channel_top': 1.0,
            'ST_tW_antitop': 1.0,
            'ST_tW_top': 1.0,
            'TTTJ': 1.0,
            'TTTT_ext1': 1.0,
            'TTTW': 1.0,
            'TTTo2L2Nu': 1.0,
            'TTToSemiLeptonic': 1.0,
            'TTWJetsToLNu': 1.0,
            'TTWW': 1.0,
            'TTZToLLNuNu_M-10': 1.0,
        },
    },
}

group['other'] = {
    'label': 'Other',
    'color': 'green',
    'scale': 1.0,
    'subsample': {
        '2018': {
            'GluGluToContinToZZTo2e2mu': 1.0,
            'GluGluToContinToZZTo2e2tau': 1.0,
            'GluGluToContinToZZTo2mu2nu': 1.0,
            'GluGluToContinToZZTo2mu2tau': 1.0,
            'GluGluToContinToZZTo4mu': 1.0,
            'GluGluToContinToZZTo4tau': 1.0,
            'WWTo2L2Nu': 1.0,
            'WZTo2L2Q': 1.0,
            'WZTo3LNu_MG': 1.0,
            'ZZTo2L2Nu': 1.0,
            'ZZTo2L2Q': 1.0,
            'ZZTo4L': 1.0,
            'WWW': 1.0,
            'WWZ': 1.0,
            'WZZ': 1.0,
            'ZZZ': 1.0,
            'tZq_ll': 1.0,
        },
    },
}

# signal
group['ggH'] = {
    'label': r'ggF H($\mu\mu$)',
    'color': 'red',
    'scale': 1.0, # stack scale
    'subsample': {
        '2018': {
            'GluGluHToMuMu_M125': 1.0,
        },
    },
    'signal': True,
    'plot_shape': 2, # True: only shape no stacked plot, False: both shape and stacked plot
    'shape_norm': 20, # shape normalization will plot the shape scaled by this factor without stacking 
}

group['VBF_H'] = {
    'label': r'VBF H($\mu\mu$)',
    'color': 'blue',
    'scale': 1.0, # stack scale
    'subsample': {
        '2018': {
            'VBFHToMuMu_M-125_withDipoleRecoil': 1.0,
        },
    },
    'signal': True,
    'plot_shape': 2, # True: only shape no stacked plot, False: both shape and stacked plot
    'shape_norm': 20, # shape normalization will plot the shape scaled by this factor without stacking 
}
