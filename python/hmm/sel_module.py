import uproot as up
import awkward as ak
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema, TreeMakerSchema
from coffea.nanoevents.methods import candidate
from coffea import lookup_tools
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
from coffea.btag_tools.btagscalefactor import BTagScaleFactor
ak.behavior.update(candidate.behavior)
from template.temp_class import ntuplize
import utils.functions as fc

import numpy as np
import time

# load file

class obj_sel(ntuplize):
    """
    Object selection
    """

    def __init__(self, fin=None, fout=None, year='2018', isdata=False):
        super().__init__(fin,fout)
        self.year = year
        self.isdata = isdata

    def run(self):
        events = NanoEventsFactory.from_root(self.input, schemaclass=NanoAODSchema).events()

        # count nevents
        npos = ak.sum(events.Generator.weight > 0)
        nneg = ak.sum(events.Generator.weight < 0)
        ntot = len(events.Generator.weight)
        # good pv
        sel_pv = events.PV.npvsGood > 0
        events = events.mask[sel_pv]

        ############
        # muons
        stime = fc.cout("start","Muon selection",entries=ak.sum((events.run!=None)))

        muons = events.Muon
        sel_mu_1 = (muons.mediumId) & (muons.pt > 15)  & (abs(muons.eta) < 2.4)
        sel_mu_2 = (abs(muons.dxy) < 0.2) & (abs(muons.dz) < 0.5)
        sel_mu_3 = (muons.pfIsoId >= 2) # loose relPFIso

        good_muon_mask = (sel_mu_1) & (sel_mu_2) & (sel_mu_3) 

        sel_good_muon = (ak.sum(good_muon_mask,axis=1) > 1)
        muons = muons.mask[sel_good_muon]
        good_muons = muons[good_muon_mask]
        events = events.mask[sel_good_muon]

        # Rochester correction
        good_muons['newpt'] , good_muons['newpt_up'] , good_muons['newpt_down'] = fc.apply_rochester_correction(good_muons)
        # Dress the muons by FSR Photons

        fc.cout("end","Muon selection",start_time=stime,entries=ak.sum(events.run!=None))

        ############
        # electrons
        stime=fc.cout("start","Electron selection")

        eles = events.Electron
        good_ele_mask = (eles.pt > 20) & (abs(eles.eta + eles.deltaEtaSC) < 2.5) & (eles.mvaFall17V2Iso_WP90)

        sel_no_ele = (ak.sum(good_ele_mask,axis=1) < 1)

        events = events.mask[sel_no_ele]

        fc.cout("end","Electron selection",start_time=stime,entries=ak.sum(events.run!=None))

        ############
        # jets
        stime = fc.cout("start", "Jet selection")

        jets = events.Jet
        # jet cleaned w.r.t. muons
        clean_jet_mask = fc.is_clean(jets, good_muons, 0.4)
        sel_cleanj_1 = (jets.pt > 25) & (abs(jets.eta) < 4.7) & (jets.isTight)
        good_jet_mask = clean_jet_mask & sel_cleanj_1
        sel_ngood_jet = ak.sum(good_jet_mask,axis=1) > 0
        # clean jets
        jets = jets.mask[sel_ngood_jet]
        good_jets = jets[good_jet_mask]
        events = events.mask[sel_ngood_jet]

        fc.cout("end","Jet selection",entries=ak.sum(events.run!=None),start_time=stime)

        ############
        # bjets 
        isData=False
        if not isData:
            pt = good_jets.pt; abseta = np.abs(good_jets.eta); flav = good_jets.hadronFlavour

            good_jets['btagSF'], good_jets['btagSF_up'], good_jets['btagSF_down']  = fc.get_btagsf(abseta, pt, flav, '2018')

        ############
        # store root file
        stime = fc.cout("start", "Snapshot")

        # reduce the events
        total_sel = ak.fill_none(events.run!=None, False)
        events = events[total_sel]
        good_muons = good_muons[total_sel]
        good_jets = good_jets[total_sel]

        # hlt
        trig_dict = {}
        for idx, ihlt in enumerate(['IsoMu24']):
            trig_dict[ihlt] = events.HLT[ihlt]

        # muonIdxG and electronIdxG are nested arrays, cannot be stored
        good_jets['muonIdxG'] = ak.count(good_jets.muonIdxG,axis=2)
        good_jets['electronIdxG'] = ak.count(good_jets.electronIdxG,axis=2)

        # check passed events
        npassed = len(events.run)
        
        fc.cout("summary", f"# passed:{npassed}")
        with up.recreate(self.output, compression=None) as fout:
            if npassed > 0:
                fout['Events'] = {
                    'basic': {
                        'run': events.run,
                        'luminosityBlock': events.luminosityBlock,
                        'event': events.event,
                    },
                    'Muon': good_muons,
                    'Jet': good_jets,
                    'MET': events.MET,
                    'PuppiMET': events.PuppiMET,
                    'HLT': trig_dict,
                }
            else:
                fc.cout('warning', "no 'Events' tree")
            fout['nEvents'] = {
                'count': {
                    'npos': np.array([npos],dtype=np.int32),
                    'nneg': np.array([nneg],dtype=np.int32),
                    'ntot': np.array([ntot],dtype=np.int32),
                }
            }
        fc.cout("end", "Snapshot", start_time=stime)

if __name__ == '__main__':
    fc.cout("this is hmm")