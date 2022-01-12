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
import numpy as np
import time


def count_time(step_name, initial_time = None, is_start = True):
    if is_start:
        print("[START]\t ",f">>> {step_name} <<<")
        return time.time()
    else:
        if initial_time == None:
            print("*ERROR*","please input the initial_time, now exit")
            exit(0)
        else:
            print("[ END ]\t ",f">>> {step_name} <<<","\t cost time:",np.round(time.time()-initial_time,2),"s")
            return 


# https://gitlab.cern.ch/akhukhun/roccor
# https://github.com/CoffeaTeam/coffea/blob/master/coffea/lookup_tools/rochester_lookup.py
# https://github.com/TopEFT/topcoffea/blob/master/topcoffea/modules/corrections.py#L359
def apply_rochester_correction(mu, is_mc=True, year='2018'):
    if year=='2016': rochester_data = lookup_tools.txt_converters.convert_rochester_file("data/MuonScale/RoccoR2016.txt", loaduncs=True)
    elif year=='2017': rochester_data = lookup_tools.txt_converters.convert_rochester_file("data/MuonScale/RoccoR2017.txt", loaduncs=True)
    elif year=='2018': rochester_data = lookup_tools.txt_converters.convert_rochester_file("/data/pubfs/xiaoj/test/roc/RoccoR/RoccoR2018.txt", loaduncs=True)
    rochester = lookup_tools.rochester_lookup.rochester_lookup(rochester_data)
    if is_mc:
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
  elif year == '2018': SFevaluatorBtag = BTagScaleFactor("/data/pubfs/xiaoj/test/DeepFlav_2018.csv","MEDIUM")
  else: raise Exception(f"Error: Unknown year \"{year}\".")

  return SFevaluatorBtag.eval("central",flavor,eta,pt), SFevaluatorBtag.eval("up",flavor,eta,pt), SFevaluatorBtag.eval("down",flavor,eta,pt)


