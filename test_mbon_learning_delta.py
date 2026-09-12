import sys, json
import numpy as np
import pandas as pd
from pathlib import Path
from brian2 import Network, StateMonitor, ms, mV
from dopamine_learning import FlyMemory, apply_learned_weights
from fly_mb_policy import FlyMBPolicy, ACTION_MBONS

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "code" / "paper-phil-drosophila"))
from model import create_model, default_params

COMP_FILE = BASE / "data" / "2025_Completeness_783.csv"
CON_FILE = BASE / "data" / "2025_Connectivity_783.parquet"

TEST_STATES = [(-1,0),(1,0),(0,-1),(0,1)]
EXPECTED = {(-1,0):"LEFT",(1,0):"RIGHT",(0,-1):"UP",(0,1):"DOWN"}

brain = FlyMBPolicy()
memory = FlyMemory()

with open(BASE / "mbon_calibration.json","r",encoding="utf-8") as f:
    calibration = json.load(f)

comp = pd.read_csv(COMP_FILE,index_col=0)
flyid2i = {int(fid):i for i,fid in enumerate(comp.index)}
ACTION_INDICES = {a:flyid2i[fid] for a,fid in ACTION_MBONS.items()}

def apply_state_calibration(syn,state):
    key = f"{state[0]},{state[1]}"
    pre = np.asarray(syn.i[:],dtype=np.int64)
    post = np.asarray(syn.j[:],dtype=np.int64)
    changed = 0
    for action in ACTION_MBONS:
        factor = float(calibration[key][action]["factor"])
        mbon = brain.action_mbon_indices[action]
        kcs = [int(c["kc_index"]) for c in brain.real_map[key][action]["connections"]]
        mask = np.isin(pre,kcs) & (post == mbon)
        if mask.any():
            syn.w[mask] = syn.w[mask] * factor
            changed += int(mask.sum())
    return changed

def run_state(state, learned):
    params = dict(default_params)
    params["t_run"] = 50*ms
    neu,syn,spike_monitor = create_model(str(COMP_FILE),str(CON_FILE),params)
    ncal = apply_state_calibration(syn,state)
    nlearn = apply_learned_weights(syn,memory) if learned else 0

    for kc in brain.get_state_kcs(state):
        neu.v[kc] = -44*mV

    mon = StateMonitor(neu,["v","g"],record=[ACTION_INDICES[a] for a in ACTION_MBONS])
    net = Network(neu,syn,spike_monitor,mon)
    net.run(params["t_run"])

    t = np.asarray(mon.t/ms,dtype=float)
    out = {}
    for row,action in enumerate(ACTION_MBONS):
        g = np.asarray(mon.g[row]/mV,dtype=float)
        v = np.asarray(mon.v[row]/mV,dtype=float)
        pg = np.maximum(g,0.0)
        integ = float(np.trapz(pg,t)) if len(t) >= 2 else 0.0
        out[action] = {
            "spikes": int(spike_monitor.count[ACTION_INDICES[action]]),
            "peak_g": float(np.max(g)),
            "integrated_g": integ,
            "peak_v": float(np.max(v)),
        }
    return out,ncal,nlearn

print("="*100)
print("LEARNED - UNLEARNED REAL MBON ACTIVITY")
print("="*100)

pc = ic = sc = 0
for state in TEST_STATES:
    expected = EXPECTED[state]
    print("\n"+"="*70)
    print("STATE:",state,"EXPECTED:",expected)
    print("="*70)

    base,ncal,_ = run_state(state,False)
    learned,_,nlearn = run_state(state,True)
    print("Calibration synapses:",ncal)
    print("Learned synapses applied:",nlearn)

    delta = {}
    for action in ACTION_MBONS:
        b,a = base[action],learned[action]
        delta[action] = {
            "spikes": a["spikes"]-b["spikes"],
            "peak_g": a["peak_g"]-b["peak_g"],
            "integrated_g": a["integrated_g"]-b["integrated_g"],
        }
        mark = "  <-- EXPECTED" if action == expected else ""
        print(
            f"{action:5s} | spikes {b['spikes']}->{a['spikes']} (Δ {delta[action]['spikes']:+d}) | "
            f"peak_g {b['peak_g']:7.3f}->{a['peak_g']:7.3f} (Δ {delta[action]['peak_g']:+7.3f}) | "
            f"int_g {b['integrated_g']:8.3f}->{a['integrated_g']:8.3f} "
            f"(Δ {delta[action]['integrated_g']:+8.3f}){mark}"
        )

    pw = max(delta,key=lambda x:delta[x]["peak_g"])
    iw = max(delta,key=lambda x:delta[x]["integrated_g"])
    sw = max(delta,key=lambda x:delta[x]["spikes"])

    print("BIGGEST ΔPEAK_G:",pw,"CORRECT" if pw==expected else "WRONG")
    print("BIGGEST ΔINTEGRATED_G:",iw,"CORRECT" if iw==expected else "WRONG")
    print("BIGGEST ΔSPIKES:",sw,"CORRECT" if sw==expected else "WRONG")

    pc += (pw==expected)
    ic += (iw==expected)
    sc += (sw==expected)

print("\n"+"="*100)
print("FINAL SUMMARY")
print("="*100)
print(f"ΔPeak-g correct: {pc}/{len(TEST_STATES)}")
print(f"ΔIntegrated-g correct: {ic}/{len(TEST_STATES)}")
print(f"ΔSpike-count correct: {sc}/{len(TEST_STATES)}")
