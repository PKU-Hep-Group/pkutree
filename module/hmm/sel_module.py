from cmath import log
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
from coffea import hist as chist

import logging
logger = logging.getLogger('sel_module')


class obj_sel(processor.ProcessorABC):
    """
    Object selection
    """

    def __init__(self, year='2018', data=False):
        self._accumulator = processor.dict_accumulator(
            {
                "ntot": processor.defaultdict_accumulator(int),
                "npos": processor.defaultdict_accumulator(int),
                "nneg": processor.defaultdict_accumulator(int),
                "neff": processor.defaultdict_accumulator(int),
                "npass": processor.defaultdict_accumulator(int),
            }
        )
        self.year = year
        self.data = data

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        result = self.accumulator.identity()
        dataset = events.metadata['dataset']
        outpath = events.metadata['outpath']
        
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
        
        # good pv
        sel_pv = events.PV.npvsGood > 0
        events = events.mask[sel_pv]


        ############
        # muons
        # logger.info(">>> Muon selection >>> entries %s",ak.sum((events.run!=None)))

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
            good_muons['is_real'] = (~np.isnan(ak.fill_none(good_muons.matched_gen.pt, np.nan)))*1
        # ordered by new pt: high -> low
        index = ak.argsort(good_muons.pt, ascending=False)
        good_muons = good_muons[index]
        
        # Dress the muons by FSR Photons

        # logger.info("<<< Muon selection <<< entries")

        ############
        # electrons
        # stime=com.cout("start","Electron selection")

        eles = events.Electron
        good_ele_mask = (eles.pt > 20) & (abs(eles.eta + eles.deltaEtaSC) < 2.5) & (eles.mvaFall17V2Iso_WP90)

        sel_no_ele = (ak.sum(good_ele_mask,axis=1) < 1)

        events = events.mask[sel_no_ele]

        # com.cout("end","Electron selection",start_time=stime,entries=ak.sum(events.run!=None))

        ############
        # jets
        # stime = com.cout("start", "Jet selection")

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
            good_jets['is_real'] = (~np.isnan(ak.fill_none(good_jets.matched_gen.pt, np.nan)))*1
            
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

        # com.cout("end","Jet selection",entries=ak.sum(events.run!=None),start_time=stime)

        ############
        # MET
        # consider the muon pt corr
        # stime = com.cout("start", "MET selection")
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
        # com.cout("end","MET selection",entries=ak.sum(events.run!=None),start_time=stime)
        
        ############
        # store root file
        # stime = com.cout("start", "Snapshot")

        # reduce the events
        total_sel = ak.fill_none(events.run!=None, False)
        events = events[total_sel]
        good_muons = good_muons[total_sel]
        good_jets = good_jets[total_sel]
        MET = MET[total_sel]

        # check passed events
        npassed = len(events.run)
        if npassed > 0:
            eve_dict = {}
            # basic info
            eve_dict['run'] = events.run
            eve_dict['luminosityBlock'] = events.luminosityBlock
            eve_dict['event'] = events.event
            # hlt
            for ihlt in ['IsoMu24']:
                eve_dict[f"HLT_{ihlt}"] = events.HLT[ihlt]*1
            eve_dict['metFilter'] = ak.Array(ana.get_metFilter(events,self.year))*1       
            # other info for MC
            if not self.data:
                eve_dict['Generator_weight'] = events.Generator.weight
                eve_dict['nLHEPdfWeight'] = ak.count(events.LHEPdfWeight,axis=-1)
                eve_dict['LHEPdfWeight'] = events.LHEPdfWeight
                eve_dict['nLHEScaleWeight'] = ak.count(events.LHEScaleWeight,axis=-1)
                eve_dict['LHEScaleWeight'] = events.LHEScaleWeight
                eve_dict['nPSWeight'] = ak.count(events.PSWeight,axis=-1)
                eve_dict['PSWeight'] = events.PSWeight
                # pu weights
                eve_dict['PUWeight_nominal'], eve_dict['PUWeight_up'], eve_dict['PUWeight_down'] = ana.get_pusf(events.Pileup.nTrueInt, self.year)
                # genpart
                eve_dict['nGenPart'] = ak.count(events.GenPart.pt,axis=-1)
                for ibr in events.GenPart.fields:
                    if not ibr.endswith('IdxMotherG') and not ibr.endswith('IdxG'):
                        eve_dict[f"GenPart_{ibr}"] = events.GenPart[ibr]
                # genjet
                eve_dict['nGenJet'] = ak.count(events.GenJet.pt,axis=-1)
                for ibr in events.GenJet.fields:
                    if ibr=='hadronFlavour':
                        eve_dict[f"GenJet_{ibr}"] = ak.values_astype(events.GenJet[ibr], np.int32)
                    else:
                        eve_dict[f"GenJet_{ibr}"] = events.GenJet[ibr]
                # GenDressedLepton
                eve_dict['nGenDressedLepton'] = ak.count(events.GenDressedLepton.pt,axis=-1)
                for ibr in events.GenDressedLepton.fields:
                    if ibr == 'hasTauAnc':
                        eve_dict[f"GenDressedLepton_{ibr}"] = events.GenDressedLepton[ibr]*1
                    else:               
                        eve_dict[f"GenDressedLepton_{ibr}"] = events.GenDressedLepton[ibr]
            # muon info
            eve_dict['nMuon'] = ak.count(good_muons.pt,axis=-1)
            for ibr in good_muons.fields:
                eve_dict[f'Muon_{ibr}'] = good_muons[ibr]
            # jet info
            eve_dict['nJet'] = ak.count(good_jets.pt,axis=-1)
            for ibr in good_jets.fields:
                if not ibr in ['muonIdxG','electronIdxG']:
                    eve_dict[f'Jet_{ibr}'] = good_jets[ibr]
            # met info
            for ibr in MET.fields:
                eve_dict[f'MET_{ibr}'] = MET[ibr]
            # puppimet info
            for ibr in events.PuppiMET.fields:
                eve_dict[f'PuppiMET_{ibr}'] = events.PuppiMET[ibr]

            eve_ak = ak.Array(eve_dict) # this will make the store step much faster
            ak.to_parquet(eve_ak,f"{outpath}/{dataset}_{com.get_randomstr()}.parquet")
        else:
            pass

        result["ntot"][dataset] += ntot
        result["npos"][dataset] += npos
        result["nneg"][dataset] += nneg
        result["neff"][dataset] += (npos-nneg)
        result["npass"][dataset] += npassed
        return result

    def postprocess(self, accumulator):
        return accumulator


class vbf_sel(processor.ProcessorABC):
    """
    VBF selection after Object selection
    """

    def __init__(self, info_dict):
        self._accumulator = processor.dict_accumulator(
            {
                "ntot": processor.defaultdict_accumulator(int),
                "npos": processor.defaultdict_accumulator(int),
                "nneg": processor.defaultdict_accumulator(int),
                "neff": processor.defaultdict_accumulator(int),
                "npass": processor.defaultdict_accumulator(int),
                "zmass": chist.Hist(
                    "Events",
                    chist.Cat("dataset", "Dataset"),
                    chist.Bin("zmass", "$m_{\mu\mu}$ [GeV]", 30, 76, 106),                 
                ),                
                "hmass": chist.Hist(
                    "Events",
                    chist.Cat("dataset", "Dataset"),
                    chist.Bin("hmass", "$m_{\mu\mu}$ [GeV]", 40, 110, 150),                 
                ),                
                "mass": chist.Hist(
                    "Events",
                    chist.Cat("dataset", "Dataset"),
                    chist.Bin("mass", "$m_{\mu\mu}$ [GeV]", 80, 70, 150),                 
                ),                
                "met_pt": chist.Hist(
                    "Events",
                    chist.Cat("dataset", "Dataset"),
                    chist.Bin("met_pt", "p$_{T}^{miss}$ [GeV]", 80, 0, 200),                 
                ),                
                "mjj": chist.Hist(
                    "Events",
                    chist.Cat("dataset", "Dataset"),
                    chist.Bin("mjj", "m$_{jj}$ [GeV]", 80, 400, 2500),                 
                ),                
            }
        )

        self.info_dict = info_dict

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        result = self.accumulator.identity()
        dataset = events.metadata['dataset']
        # get the info
        data = self.info_dict[dataset].get("data", False)
        signal = self.info_dict[dataset].get("signal", False)
        datadriven = self.info_dict[dataset].get("datadriven", False)
        norm = self.info_dict[dataset].get("norm", 1.)
        year = self.info_dict[dataset].get("year", "2018")
        weight = self.info_dict[dataset].get("weight", "1")
        brachmap = self.info_dict[dataset].get("brachmap", {})

        # handle the nuisance
        # weight
        dummy_weight = np.ones(len(events))
        if not data:
            if hasattr(events, "LHEPdfWeight"):
                LHEPdfWeight = events.LHEPdfWeight
            if hasattr(events, "LHEScaleWeight"):
                LHEScaleWeight = events.LHEScaleWeight
            if hasattr(events, "PSWeight"):
                PSWeight = events.PSWeight
            if hasattr(events, "PUWeight"):
                PUWeight = events.PUWeight
        try:
            additional_weight = eval("dummy_weight*{}".format(weight))
        except:
            logger.error("Invalid weight expression: %s", weight)
            exit(1)
        # shape
        muons = events.Muon
        jets = events.Jet
        MET = events.MET
        if len(brachmap) > 0:
            for ibr in brachmap:
                if ibr.startswith("Muon_"):
                    muons[ibr[5:]] = muons[brachmap[ibr][5:]]
                if ibr.startswith("Jet_"):
                    jets[ibr[4:]] = jets[brachmap[ibr][4:]]
                if ibr.startswith("MET_"):
                    MET[ibr[4:]] = MET[brachmap[ibr][4:]]
                if not data:
                    if ibr.startswith("PUWeight_"):
                        PUWeight[ibr[9:]] = PUWeight[brachmap[ibr][9:]]
        # ordered by new pt: high -> low
        index = ak.argsort(muons.pt, ascending=False)
        muons = muons[index]
        index = ak.argsort(jets.pt, ascending=False)
        jets = jets[index]

        # get the number of events
        if not data:
            # count nevents
            npos = ak.sum(events.Generator.weight > 0)
            nneg = ak.sum(events.Generator.weight < 0)
            ntot = len(events.Generator.weight)
            norm *= np.sign(events.Generator.weight)
        else:
            npos = len(events)
            nneg = 0
            ntot = len(events)            
            norm *= np.ones(len(events))

        # print(events.fields)
        # print(type(events))
        ############
        # trigger
        events = events.mask[events.HLT.IsoMu24 > 0.5]
        ############
        # muons
        # logger.info(">>> Muon selection >>> entries %s",ak.sum((events.run!=None)))
        sel_nmu = ak.count(muons.pt,axis=-1) == 2
        events = events.mask[sel_nmu]
        align_sel = ak.fill_none(events.run!=None, False)
        muons = muons.mask[align_sel]
        if not year == "2017":
            sel_mu_1 = (muons.pt[:,0] > 26) & (muons.pt[:,1] > 20)
        else:
            sel_mu_1 = (muons.pt[:,0] > 29) & (muons.pt[:,1] > 20)
        if not data:
            sel_mu_2 = (muons.is_real[:,0] > 0.5) & (muons.is_real[:,1] > 0.5)
            sel_mu_tot = sel_mu_1 & sel_mu_2
        else:
            sel_mu_tot = sel_mu_1
        muons = muons.mask[sel_mu_tot]
        events = events.mask[sel_mu_tot]
        
        # ############
        # # jets
        # # logger.info(">>> Jet selection >>> entries %s",ak.sum((events.run!=None)))
        align_sel = ak.fill_none(events.run!=None, False)
        jets = jets.mask[align_sel]
        # bjet
        medium_bjet = (jets.btagDeepFlavB > 0.2770) & (jets.pt > 25) & (abs(jets.eta) < 2.4)
        n_medium_bjet = ak.sum(medium_bjet,axis=1)
        loose_bjet = (jets.btagDeepFlavB > 0.0494) & (jets.pt > 25) & (abs(jets.eta) < 2.4)
        n_loose_bjet = ak.sum(loose_bjet,axis=1)
        jets = jets.mask[(n_medium_bjet < 1) & (n_loose_bjet < 2)]

        ljet_tag = (jets.btagDeepFlavB <= 0.0494)
        n_ljet = ak.sum(ljet_tag,axis=1)
        jets = jets.mask[n_ljet >= 2]
        ljets = jets[ljet_tag]

        # vbs cuts
        sel_ljet_1 = (ljets.pt[:,0] > 35) & (ljets.pt[:,1] > 25)
        sel_ljet_2 = ((ljets[:,0] + ljets[:,1]).mass > 400) & (np.abs(ljets[:,0].eta - ljets[:,1].eta) > 2.5)

        ljets = ljets.mask[sel_ljet_1 & sel_ljet_2]
        events = events.mask[sel_ljet_1 & sel_ljet_2]

        ########
        # Total
        total_sel = ak.fill_none(events.run!=None, False)
        events = events[total_sel]
        muons = muons[total_sel]
        ljets = ljets[total_sel]
        MET = MET[total_sel]
        wgt = norm[total_sel]*additional_weight[total_sel]

        # check passed events
        npassed = len(events.run)

        ############
        # fill histograms
        dimuon_mass = (muons[:,0] + muons[:,1]).mass
        result["zmass"].fill(
            dataset=dataset,
            zmass=dimuon_mass,
            weight = wgt,
        )
        result["hmass"].fill(
            dataset=dataset,
            hmass=dimuon_mass,
            weight = wgt,
        )
        result["mass"].fill(
            dataset=dataset,
            mass=dimuon_mass,
            weight = wgt,
        )
        result["met_pt"].fill(
            dataset=dataset,
            met_pt=MET.pt,
            weight = wgt,
        )
        result["mjj"].fill(
            dataset=dataset,
            mjj=(ljets[:,0] + ljets[:,1]).mass,
            weight = wgt,
        )
        result["ntot"][dataset] += ntot
        result["npos"][dataset] += npos
        result["nneg"][dataset] += nneg
        result["neff"][dataset] += (npos-nneg)
        result["npass"][dataset] += npassed
        return result

    def postprocess(self, accumulator):
        return accumulator
