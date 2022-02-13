# example of configuration file
date = '220211'

tag = "_".join(['hmm',date])

# file with list of samples
samples_cfg = {
    '2018': 'samples2018.py',
}
# nuisances file for mkDatacards and for mkShape
nuisances_cfg = {
    '2018': 'nuisances2018.py',
} 
# file with plot configurations
plot_cfg = 'plots.py'

# output of the histograms
out_hist = "_".join(['hist',tag])
# used by mkPlot to define output directory for plots
# different from "outputDir" to do things more tidy
out_plot = "_".join(['plot',tag])



