import uproot as up
import awkward as ak
import coffea
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema, TreeMakerSchema
from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea import lookup_tools
from coffea.lookup_tools import extractor
from coffea.jetmet_tools import FactorizedJetCorrector, JetCorrectionUncertainty
from coffea.jetmet_tools import JECStack, CorrectedJetsFactory, CorrectedMETFactory
from coffea.btag_tools.btagscalefactor import BTagScaleFactor
ak.behavior.update(candidate.behavior)
from template.temp_class import ntuplize
import numpy as np
import argparse
import os
from utils import common_helper as com
from utils import analyze_helper as ana

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('-i','--input', help='input file', default=None)
parser.add_argument('-o','--outdir', help='output folder, not the file name', default=None)
parser.add_argument('-y','--year', help='which year', default='2018')
parser.add_argument('-d','--data', help='is data or mc, default: False', action='store_true', default=False)
parser.add_argument('-s','--step', help='which step', default='obj_sel')
parser.add_argument('-sp','--sample', help='sample name', default=None)
args = parser.parse_args()

class obj_sel(processor.ProcessorABC):
    """
    Object selection
    """

    def __init__(self, outdir, subdir='obj_sel', year='2018', data=False):
        self._accumulator = processor.dict_accumulator(
            {
                "ntot": processor.defaultdict_accumulator(float),
                "npos": processor.defaultdict_accumulator(float),
                "nneg": processor.defaultdict_accumulator(float),
            }
        )
        self.outdir = outdir
        self.subdir = subdir
        self.year = year
        self.data = data

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        result = self.accumulator.identity()
        dataset = events.metadata['dataset']

        if not self.data:
            # count nevents
            npos = ak.sum(events.Generator.weight > 0)
            nneg = ak.sum(events.Generator.weight < 0)
            ntot = len(events.Generator.weight)
        else:
            npos = len(events)
            nneg = 0
            ntot = len(events)
            # lumi mask
            lumi_mask = ana.get_lumi_mask(events, self.year)
            events = events.mask[lumi_mask]

        result["ntot"][dataset] += ntot
        result["npos"][dataset] += npos
        result["nneg"][dataset] += nneg
        
        # good pv
        sel_pv = events.PV.npvsGood > 0
        events = events.mask[sel_pv]


        ############
        # muons
        stime = com.cout("start","Muon selection",entries=ak.sum((events.run!=None)))

        muons = events.Muon
        sel_mu_1 = (muons.mediumId) & (muons.pt > 15)  & (abs(muons.eta) < 2.4)
        sel_mu_2 = (abs(muons.dxy) < 0.2) & (abs(muons.dz) < 0.5)
        sel_mu_3 = (muons.pfIsoId >= 2) # loose relPFIso

        good_muon_mask = (sel_mu_1) & (sel_mu_2) & (sel_mu_3) 

        sel_good_muon = (ak.sum(good_muon_mask,axis=1) > 1)
        muons = muons.mask[sel_good_muon]
        good_muons = muons[good_muon_mask]
        events = events.mask[sel_good_muon]

        good_muons['pt_orig'] = good_muons.pt
        # Rochester correction, the nomial pt will be updated
        good_muons['pt'] , good_muons['pt_roccor_up'] , good_muons['pt_roccor_down'] = ana.apply_rochester_correction(good_muons,self.data,self.year)
        if not self.data:
            good_muons['is_real'] = ~np.isnan(ak.fill_none(good_muons.matched_gen.pt, np.nan))
        # ordered by new pt: high -> low
        index = ak.argsort(good_muons.pt, ascending=False)
        good_muons = good_muons[index]
        
        # Dress the muons by FSR Photons

        com.cout("end","Muon selection",start_time=stime,entries=ak.sum(events.run!=None))

        ############
        # electrons
        stime=com.cout("start","Electron selection")

        eles = events.Electron
        good_ele_mask = (eles.pt > 20) & (abs(eles.eta + eles.deltaEtaSC) < 2.5) & (eles.mvaFall17V2Iso_WP90)

        sel_no_ele = (ak.sum(good_ele_mask,axis=1) < 1)

        events = events.mask[sel_no_ele]

        com.cout("end","Electron selection",start_time=stime,entries=ak.sum(events.run!=None))

        ############
        # jets
        stime = com.cout("start", "Jet selection")

        jets = events.Jet
        # jet cleaned w.r.t. muons
        clean_jet_mask = ana.is_clean(jets, good_muons, 0.4)
        sel_cleanj_1 = (jets.pt > 25) & (abs(jets.eta) < 4.7) & (jets.isTight)
        good_jet_mask = clean_jet_mask & sel_cleanj_1
        sel_ngood_jet = ak.sum(good_jet_mask,axis=1) > 0
        # clean jets
        jets = jets.mask[sel_ngood_jet]
        good_jets = jets[good_jet_mask]
        events = events.mask[sel_ngood_jet]

        # jesr
        good_jets['pt_orig'] = good_jets.pt
        good_jets['mass_orig'] = good_jets.mass
        if not self.data:
            good_jets['is_real'] = ~np.isnan(ak.fill_none(good_jets.matched_gen.pt, np.nan))
            
            good_jets["pt_raw"] = (1 - good_jets.rawFactor)*good_jets.pt
            good_jets["mass_raw"] = (1 - good_jets.rawFactor)*good_jets.mass
            good_jets["pt_gen"] = ak.values_astype(ak.fill_none(good_jets.matched_gen.pt, 0), np.float32)
            good_jets["rho"] = ak.broadcast_arrays(events.fixedGridRhoFastjetAll, good_jets.pt)[0]
            events_cache = events.caches[0]
            corrected_jets = ana.apply_jet_corrections('2018').build(good_jets, lazy_cache=events_cache)
            jesr_unc = [i for i in corrected_jets.fields if i.startswith("JES") or i.startswith("JER")]
            good_jets["pt"] = corrected_jets.pt
            good_jets["mass"] = corrected_jets.mass
            for ibr in jesr_unc:
                good_jets[f"pt_{ibr}_up"] = corrected_jets[ibr].up.pt
                good_jets[f"pt_{ibr}_down"] = corrected_jets[ibr].down.pt
                good_jets[f"mass_{ibr}_up"] = corrected_jets[ibr].up.mass
                good_jets[f"mass_{ibr}_down"] = corrected_jets[ibr].down.mass
            # ordered by new pt: high -> low
            index = ak.argsort(good_jets.pt, ascending=False)
            good_jets = good_jets[index]
            
        ############
        # bjets 
        if not self.data:
            flav = good_jets.hadronFlavour; abseta = np.abs(good_jets.eta); pt = good_jets.pt

            good_jets['btagSF'], good_jets['btagSF_up'], good_jets['btagSF_down']  = ana.get_btagsf(flav, abseta, pt, '2018')

        com.cout("end","Jet selection",entries=ak.sum(events.run!=None),start_time=stime)

        ############
        # MET
        # consider the muon pt corr
        stime = com.cout("start", "MET selection")
        MET = events.MET
        MET['pt_orig'] = MET.pt
        MET['phi_orig'] = MET.phi
        MET['pt_roccor'], MET['phi_roccor'] = ana.corrected_polar_met(MET.pt,MET.phi,good_muons.pt,good_muons.phi,good_muons.pt_orig)
        # consider the jer corr, please note: for jets, the pt_raw is the pt_orig, think about it
        if not self.data:
            # the jer is applied after considering roccorr on Muon
            MET['pt'], MET['phi'] = ana.corrected_polar_met(MET['pt_roccor'],MET['phi_roccor'],good_jets["pt"],good_jets["phi"],good_jets["pt_orig"])
            # uncertainties
            MET['pt_roccor_up'], MET['phi_roccor_up'] = ana.corrected_polar_met(MET.pt,MET.phi,good_muons.pt_roccor_up,good_muons.phi,good_muons.pt)
            MET['pt_roccor_down'], MET['phi_roccor_down'] = ana.corrected_polar_met(MET.pt,MET.phi,good_muons.pt_roccor_down,good_muons.phi,good_muons.pt)
            MET['pt_UnclusteredEnergy_up'], MET['phi_UnclusteredEnergy_up'] = ana.corrected_polar_met(
                MET['pt'],
                MET['phi'],
                good_jets["pt"],
                good_jets["phi"],
                good_jets["pt"],
                (
                    True,
                    MET.MetUnclustEnUpDeltaX,
                    MET.MetUnclustEnUpDeltaY,
                ),
            )
            MET['pt_UnclusteredEnergy_down'], MET['phi_UnclusteredEnergy_down'] = ana.corrected_polar_met(
                MET['pt'],
                MET['phi'],
                good_jets["pt"],
                good_jets["phi"],
                good_jets["pt"],
                (
                    False,
                    MET.MetUnclustEnUpDeltaX,
                    MET.MetUnclustEnUpDeltaY,
                ),
            )
            for ibr in jesr_unc:
                MET[f"pt_{ibr}_up"], MET[f"phi_{ibr}_up"] = ana.corrected_polar_met(MET['pt'],MET['phi'],good_jets[f"pt_{ibr}_up"],good_jets["phi"],good_jets["pt"])
                MET[f"pt_{ibr}_down"], MET[f"phi_{ibr}_down"] = ana.corrected_polar_met(MET['pt'],MET['phi'],good_jets[f"pt_{ibr}_down"],good_jets["phi"],good_jets["pt"])
        com.cout("end","MET selection",entries=ak.sum(events.run!=None),start_time=stime)
        
        ############
        # store root file
        stime = com.cout("start", "Snapshot")

        # reduce the events
        total_sel = ak.fill_none(events.run!=None, False)
        events = events[total_sel]
        good_muons = good_muons[total_sel]
        good_jets = good_jets[total_sel]
        MET = MET[total_sel]
        # hlt
        trig_dict = {}
        for idx, ihlt in enumerate(['IsoMu24']):
            trig_dict[f"hlt_{ihlt}"] = events.HLT[ihlt]
        trig_dict['metFilter'] = ana.get_metFilter(events,self.year)

        # pu weights
        if not self.data:
            pu_wgt, pu_wgt_up, pu_wgt_down = ana.get_pusf(events.Pileup.nTrueInt, self.year)

        # muonIdxG and electronIdxG are nested arrays, cannot be stored
        good_jets['muonIdxG'] = ak.count(good_jets.muonIdxG,axis=2)
        good_jets['electronIdxG'] = ak.count(good_jets.electronIdxG,axis=2)

        # check passed events
        npassed = len(events.run)
        
        com.cout("summary", f"# passed:{npassed}")
        with up.recreate(f"{self.outdir}/{dataset}_{com.get_randomstr()}.root", compression=None) as fout:
            if npassed > 0:
                if self.data:
                    fout['Events'] = {
                        'basic': {
                            'run': events.run,
                            'luminosityBlock': events.luminosityBlock,
                            'event': events.event,
                        },
                        'Muon': good_muons,
                        'Jet': good_jets,
                        'MET': MET,
                        'PuppiMET': events.PuppiMET,
                        'Flag': trig_dict,
                    }
                else:
                    fout['Events'] = {
                        'basic': {
                            'run': events.run,
                            'luminosityBlock': events.luminosityBlock,
                            'event': events.event,
                        },
                        'LHEPdf': {'Weight': events.LHEPdfWeight},
                        'LHEScale': {'Weight': events.LHEPdfWeight},
                        'PS': {'Weight': events.PSWeight},
                        'PUWeight': {
                            'nominal': pu_wgt,
                            'up': pu_wgt_up,
                            'down': pu_wgt_down,
                        },
                        'Muon': good_muons,
                        'Jet': good_jets,
                        'MET': MET,
                        'PuppiMET': events.PuppiMET,
                        'Flag': trig_dict,
                    }
            else:
                com.cout('warning', "no 'Events' tree")
            fout['nEvents'] = {
                'count': {
                    'npos': np.array([npos],dtype=np.int32),
                    'nneg': np.array([nneg],dtype=np.int32),
                    'ntot': np.array([ntot],dtype=np.int32),
                    'neff': np.array([npos-nneg],dtype=np.int32),
                    'npass': np.array([npassed],dtype=np.int32),
                }
            }
        com.cout("end", "Snapshot", start_time=stime)
        return result

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
    
    ### single file for condor job
    if args.input and args.sample:
        com.cout("Single file mode!!!")
        events = NanoEventsFactory.from_root(args.input, metadata={"dataset": args.sample}, schemaclass=NanoAODSchema).events()
        if args.step == "obj_sel":
            sel = obj_sel(args.outdir, args.year, args.data)
            out = sel.process(events)
            com.cout(out)
        else:
            com.cout("Please choose a valid step")
            exit(0)
    else:
        com.cout("No valid input or sample name!!!")