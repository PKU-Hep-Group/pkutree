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
from coffea.lumi_tools import LumiMask
ak.behavior.update(candidate.behavior)
import numpy as np
import utils.common_helper as hcom
import os
import data
import yaml

import logging
logger = logging.getLogger('analyze_helper')

def get_lumi_mask(events, year="2018"):
    if year == "2016" or year == "2016APV":
        golden_json_path = hcom.abs_path("data/goldenJSON/Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt")
    elif year == "2017":
        golden_json_path = hcom.abs_path("data/goldenJSON/Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt")
    elif year == "2018":
        golden_json_path = hcom.abs_path("data/goldenJSON/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt")
    else:
        raise ValueError(f"Error: Unknown year \"{year}\".")
    lumi_mask = LumiMask(golden_json_path)(events.run,events.luminosityBlock)
    return lumi_mask


# https://gitlab.cern.ch/akhukhun/roccor
# https://github.com/CoffeaTeam/coffea/blob/master/coffea/lookup_tools/rochester_lookup.py
# https://github.com/TopEFT/topcoffea/blob/master/topcoffea/modules/corrections.py#L359
def apply_rochester_correction(mu, data=False, year='2018'):
    rochester_data = lookup_tools.txt_converters.convert_rochester_file(hcom.abs_path("data/RoccoR/RoccoR2018.txt"), loaduncs=True)
 
    if year=='2016': rochester_data = lookup_tools.txt_converters.convert_rochester_file("data/MuonScale/RoccoR2016.txt", loaduncs=True)
    elif year=='2017': rochester_data = lookup_tools.txt_converters.convert_rochester_file("data/MuonScale/RoccoR2017.txt", loaduncs=True)
    elif year=='2018': rochester_data = lookup_tools.txt_converters.convert_rochester_file(hcom.abs_path("data/RoccoR/RoccoR2018.txt"), loaduncs=True)
    rochester = lookup_tools.rochester_lookup.rochester_lookup(rochester_data)
    if not data:
        hasgen = ~np.isnan(ak.fill_none(mu.matched_gen.pt, np.nan))
        mc_rand = np.random.rand(*ak.to_numpy(ak.flatten(mu.pt)).shape)
        mc_rand = ak.unflatten(mc_rand, ak.num(mu.pt, axis=1))
        corrections = np.array(ak.flatten(ak.ones_like(mu.pt)))
        errors = np.array(ak.flatten(ak.ones_like(mu.pt)))
        
        mc_kspread = rochester.kSpreadMC(mu.charge[hasgen],mu.pt[hasgen],mu.eta[hasgen],mu.phi[hasgen],mu.matched_gen.pt[hasgen])
        mc_ksmear = rochester.kSmearMC(mu.charge[~hasgen],mu.pt[~hasgen],mu.eta[~hasgen],mu.phi[~hasgen],mu.nTrackerLayers[~hasgen],mc_rand[~hasgen])
        errspread = rochester.kSpreadMCerror(mu.charge[hasgen],mu.pt[hasgen],mu.eta[hasgen],mu.phi[hasgen],mu.matched_gen.pt[hasgen])
        errsmear = rochester.kSmearMCerror(mu.charge[~hasgen],mu.pt[~hasgen],mu.eta[~hasgen],mu.phi[~hasgen],mu.nTrackerLayers[~hasgen],mc_rand[~hasgen])
        hasgen_flat = np.array(ak.flatten(hasgen))
        corrections[hasgen_flat] = np.array(ak.flatten(mc_kspread))
        corrections[~hasgen_flat] = np.array(ak.flatten(mc_ksmear))
        errors[hasgen_flat] = np.array(ak.flatten(errspread))
        errors[~hasgen_flat] = np.array(ak.flatten(errsmear))
        corrections = ak.unflatten(corrections, ak.num(mu.pt, axis=1))
        errors = ak.unflatten(errors, ak.num(mu.pt, axis=1))
    else:
        corrections = rochester.kScaleDT(mu.charge, mu.pt, mu.eta, mu.phi)
        errors = rochester.kScaleDTerror(mu.charge, mu.pt, mu.eta, mu.phi)
    
    pt_nom = mu.pt * corrections
    pt_err = mu.pt * errors
    return pt_nom, pt_nom + pt_err, pt_nom - pt_err


def is_clean(obj_A, obj_B, drmin=0.4):
    ## Method 1
    # pair_obj = ak.cartesian([obj_A, obj_B],nested=True)
    # obj1, obj2 = ak.unzip(pair_obj)
    # dr_jm = obj1.delta_r(obj2)
    # min_dr_jm = ak.min(dr_jm,axis=2)
    # mask = min_dr_jm > drmin
    
    ## Method 2
    objB_near, objB_dr = obj_A.nearest(obj_B, return_metric=True)
    mask = ak.fill_none(objB_dr > drmin, True) # I guess to use True is because if there are no objB, all the objA are clean
    return (mask)


def get_btagsf(flavor, eta, pt, sys='nominal', year='2018'):
    # Efficiencies and SFs for UL only available for 2016APV, 2017 and 2018
    # light flavor SFs and unc. missed for 2016APV
    if   (year == '2016' or year == '2016APV'): SFevaluatorBtag = BTagScaleFactor("data/btagSF/DeepFlav_2016.csv","MEDIUM") 
    elif year == '2017': SFevaluatorBtag = BTagScaleFactor("data/btagSF/UL/DeepJet_UL17.csv","MEDIUM")
    elif year == '2018': SFevaluatorBtag = BTagScaleFactor(hcom.abs_path("data/btagSF/2018/DeepJet_102XSF_V2.btag.csv.gz"),"MEDIUM")
    else: raise Exception(f"Error: Unknown year \"{year}\".")

    return SFevaluatorBtag.eval("central",flavor,eta,pt), SFevaluatorBtag.eval("up",flavor,eta,pt), SFevaluatorBtag.eval("down",flavor,eta,pt)


### JES JER
def apply_jet_corrections(year):
    extract = extractor()
    if year=='2016':
        extract.add_weight_sets(
            [
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L1FastJet_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L2Relative_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L3Absolute_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L2L3Residual_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Regrouped_Autumn18_V19_MC_UncertaintySources_AK4PFchs.junc.txt')}",
                f"* * {hcom.abs_path('data/jer/2018/Autumn18_V7b_MC_PtResolution_AK4PFchs.jr.txt')}",
                f"* * {hcom.abs_path('data/jer/2018/Autumn18_V7b_MC_SF_AK4PFchs.jersf.txt')}",
            ]
        )
    elif year=='2017':
        extract.add_weight_sets(
            [
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L1FastJet_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L2Relative_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L3Absolute_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L2L3Residual_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Regrouped_Autumn18_V19_MC_UncertaintySources_AK4PFchs.junc.txt')}",
                f"* * {hcom.abs_path('data/jer/2018/Autumn18_V7b_MC_PtResolution_AK4PFchs.jr.txt')}",
                f"* * {hcom.abs_path('data/jer/2018/Autumn18_V7b_MC_SF_AK4PFchs.jersf.txt')}",
            ]
        )
    elif year=='2018':
        extract.add_weight_sets(
            [
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L1FastJet_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L2Relative_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L3Absolute_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Autumn18_V19_MC_L2L3Residual_AK4PFchs.jec.txt')}",
                f"* * {hcom.abs_path('data/jes/2018/Regrouped_Autumn18_V19_MC_UncertaintySources_AK4PFchs.junc.txt')}",
                f"* * {hcom.abs_path('data/jer/2018/Autumn18_V7b_MC_PtResolution_AK4PFchs.jr.txt')}",
                f"* * {hcom.abs_path('data/jer/2018/Autumn18_V7b_MC_SF_AK4PFchs.jersf.txt')}",
            ]
        )
    else:
        exit(0)

    extract.finalize()
    evaluator = extract.make_evaluator()

    jec_names = dir(evaluator)
    # print(jec_names)
    jec_inputs = {name: evaluator[name] for name in jec_names}
    jec_stack = JECStack(jec_inputs)
    name_map = jec_stack.blank_name_map
    name_map['JetPt'] = 'pt'
    name_map['JetMass'] = 'mass'
    name_map['JetEta'] = 'eta'
    name_map['JetPhi'] = 'phi'
    name_map['JetA'] = 'area'
    name_map['ptGenJet'] = 'pt_gen'
    name_map['ptRaw'] = 'pt_raw'
    name_map['massRaw'] = 'mass_raw'
    name_map['Rho'] = 'rho'
    return CorrectedJetsFactory(name_map, jec_stack)


# follow: https://github.com/CoffeaTeam/coffea/blob/master/coffea/jetmet_tools/CorrectedMETFactory.py
def corrected_polar_met(met_pt, met_phi, obj_pt, obj_phi, obj_pt_orig, deltas=None):
    sj, cj = np.sin(obj_phi), np.cos(obj_phi)
    x = met_pt * np.cos(met_phi) + ak.sum(
        obj_pt * cj - obj_pt_orig * cj, axis=1
    )
    y = met_pt * np.sin(met_phi) + ak.sum(
        obj_pt * sj - obj_pt_orig * sj, axis=1
    )
    if deltas:
        positive, dx, dy = deltas
        x = x + dx if positive else x - dx
        y = y + dy if positive else y - dy
    
    # return ak.zip({"pt": np.hypot(x, y), "phi": np.arctan2(y, x)})
    return np.hypot(x, y), np.arctan2(y, x)


def get_metFilter(events, year='2018'):
    # 2018: goodVertices globalSuperTightHalo2016Filter HBHENoiseFilter HBHENoiseIsoFilter EcalDeadCellTriggerPrimitiveFilter BadPFMuonFilter ecalBadCalibFilterV2
    if year == '2016':
        metFilter = (
            events.Flag.goodVertices & \
            events.Flag.globalSuperTightHalo2016Filter & \
            events.Flag.HBHENoiseFilter & \
            events.Flag.HBHENoiseIsoFilter & \
            events.Flag.EcalDeadCellTriggerPrimitiveFilter & \
            events.Flag.BadPFMuonFilter
        )
    elif year == '2017':
        metFilter = (
            events.Flag.goodVertices & \
            events.Flag.globalSuperTightHalo2016Filter & \
            events.Flag.HBHENoiseFilter & \
            events.Flag.HBHENoiseIsoFilter & \
            events.Flag.EcalDeadCellTriggerPrimitiveFilter & \
            events.Flag.BadPFMuonFilter & \
            events.Flag.ecalBadCalibFilterV2
        )
    else:
        metFilter = (
            events.Flag.goodVertices & \
            events.Flag.globalSuperTightHalo2016Filter & \
            events.Flag.HBHENoiseFilter & \
            events.Flag.HBHENoiseIsoFilter & \
            events.Flag.EcalDeadCellTriggerPrimitiveFilter & \
            events.Flag.BadPFMuonFilter & \
            events.Flag.ecalBadCalibFilterV2
        )
    return metFilter


def get_pusf(nTrueInt, year):
    ###### Pileup reweighing
    ## 
    ####################################################
    ## Places to find UL files:
    ## NanoAOD tools: https://github.com/cms-nanoAOD/nanoAOD-tools/tree/master/python/postprocessing/data/pileup
    ## Get central PU data and MC profiles and calculate reweighting
    ## Using the current UL recommendations in:
    ##   https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupJSONFileforData
    ##   - 2018: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions18/13TeV/PileUp/UltraLegacy/
    ##   - 2017: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions17/13TeV/PileUp/UltraLegacy/
    ##   - 2016: /afs/cern.ch/cms/CAF/CMSCOMM/COMM_DQM/certification/Collisions16/13TeV/PileUp/UltraLegacy/
    ##
    ## MC histograms from:
    ##    https://github.com/CMS-LUMI-POG/PileupTools/

    mc_file_dict = {
        '2016': hcom.abs_path('data/pileup/PreUL13TeV/pileup_2016BF.root'),
        '2017': hcom.abs_path('data/pileup/PreUL13TeV/pileup_2017_shifts.root'),
        '2018': hcom.abs_path('data/pileup/PreUL13TeV/pileup_2018_shifts.root'),
    }
    data_file_dict = {
        '2016': {
            'down':     hcom.abs_path('data/pileup/PreUL13TeV/PileupHistogram-goldenJSON-13tev-2016-66000ub.root'),      
            'nominal':  hcom.abs_path('data/pileup/PreUL13TeV/PileupHistogram-goldenJSON-13tev-2016-69200ub.root'),
            'up':       hcom.abs_path('data/pileup/PreUL13TeV/PileupHistogram-goldenJSON-13tev-2016-72400ub.root'),      
        },
        '2017': {
            'down':     hcom.abs_path('data/pileup/PreUL13TeV/PileupHistogram-goldenJSON-13tev-2017-66000ub-99bins.root'),      
            'nominal':  hcom.abs_path('data/pileup/PreUL13TeV/PileupHistogram-goldenJSON-13tev-2017-69200ub-99bins.root'),
            'up':       hcom.abs_path('data/pileup/PreUL13TeV/PileupHistogram-goldenJSON-13tev-2017-72400ub-99bins.root'),      
        },
        '2018': {
            'down':     hcom.abs_path('data/pileup/PreUL13TeV/PileupHistogram-goldenJSON-13tev-2018-66000ub-99bins.root'),      
            'nominal':  hcom.abs_path('data/pileup/PreUL13TeV/PileupHistogram-goldenJSON-13tev-2018-69200ub-99bins.root'),
            'up':       hcom.abs_path('data/pileup/PreUL13TeV/PileupHistogram-goldenJSON-13tev-2018-72400ub-99bins.root'),      
        },
    }
    PUfunc = {}
    with up.open(mc_file_dict[year]) as fMC:
        hMC = fMC['pileup']
        PUfunc['MC'] = lookup_tools.dense_lookup.dense_lookup(hMC .values(), hMC.axis(0).edges())
    with up.open(data_file_dict[year]['nominal']) as fData:
        hD   = fData  ['pileup']
        PUfunc['Data'  ] = lookup_tools.dense_lookup.dense_lookup(hD  .values(), hD.axis(0).edges())
    with up.open(data_file_dict[year]['up']) as fDataUp:
        hDUp = fDataUp['pileup']
        PUfunc['DataUp'] = lookup_tools.dense_lookup.dense_lookup(hDUp.values(), hD.axis(0).edges())
    with up.open(data_file_dict[year]['down']) as fDataDo:
        hDDo = fDataDo['pileup']
        PUfunc['DataDo'] = lookup_tools.dense_lookup.dense_lookup(hDDo.values(), hD.axis(0).edges())

    if year not in ['2016','2017','2018']: raise Exception(f"Error: Unknown year \"{year}\".")
    nMC  =PUfunc['MC'](nTrueInt+1)
    nData =PUfunc['Data'](nTrueInt)
    nData_up = PUfunc['DataUp'](nTrueInt)
    nData_down = PUfunc['DataDo'](nTrueInt)
    return np.divide(nData,nMC), np.divide(nData_up,nMC), np.divide(nData_down,nMC)
