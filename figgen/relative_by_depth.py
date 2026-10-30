import json, numpy as np
from pathlib import Path
DATA = Path.home()/"rosetta_data"/"paper_n250"
C = ["agency","authorization","causation","certainty","credibility","deception","exfiltration","formality",
     "moral_valence","negation","plurality","sarcasm","sentiment","specificity","temporal_order","threat_severity","urgency"]
models=[d.name for d in sorted(DATA.iterdir()) if d.is_dir() and not d.name.startswith("_") and "_it" not in d.name[-3:] and "Instruct" not in d.name
        and all((d/f"caz_{c}.json").exists() for c in C)]
fam=lambda m: next(f for f,k in [("pythia","pythia"),("gptneo","gpt_neo"),("qwen","Qwen"),("opt","opt"),("gemma","gemma"),("llama","Llama"),("phi","phi"),("mistral","istral"),("gpt2","gpt2"),("falcon","falcon")] if k in m)
# load per-layer unit vectors and separations: V[m][c] = (L x d), S[m][c] = (L,)
V={}; S={}
for m in models:
    V[m]={}; S[m]={}
    for c in C:
        ld=json.load(open(DATA/m/f"caz_{c}.json"))["layer_data"]
        M=np.array([x["dom_vector"] for x in ld["metrics"]],float); M/=np.linalg.norm(M,axis=1,keepdims=True)+1e-12
        V[m][c]=M; S[m][c]=np.array([x["separation_fisher"] for x in ld["metrics"]])
def gram_at(m,frac):
    L=V[m][C[0]].shape[0]; l=int(round(frac*(L-1)))
    M=np.vstack([V[m][c][l] for c in C]); return M@M.T, l, L
def lofo_acc(frac, anchors=None):
    X={}
    for m in models:
        Gm,_,_=gram_at(m,frac)
        X[m]=np.array([np.delete(Gm[j],j) for j in range(17)]) if anchors is None else np.array([[Gm[a,j] for a in anchors] for j in range(17) if j not in anchors])
    correct=total=0
    for m in models:
        cent=np.mean([X[o] for o in models if fam(o)!=fam(m)],axis=0)
        for k in range(X[m].shape[0]): correct+=(np.argmin(np.linalg.norm(cent-X[m][k],axis=1))==k); total+=1
    return correct/total
ia=C.index("sentiment")
print(f"{len(models)} base models. Concept identification from anchor-relative coordinates, leave-one-FAMILY-out, by relative depth:\n")
print(f"{'depth':>6s} {'full profile':>13s} {'sentiment-anchor only':>22s} {'median sentiment sep.':>22s} {'within/between (sent. anchor)':>30s}")
for frac in [0.0,0.1,0.25,0.4,0.5,0.6,0.75,0.9,1.0]:
    a_full=lofo_acc(frac); a_one=lofo_acc(frac,[ia])
    seps=[S[m]["sentiment"][gram_at(m,frac)[1]] for m in models]
    cols=np.array([[gram_at(m,frac)[0][ia,j] for j in range(17) if j!=ia] for m in models])  # models x 16
    within=cols.std(axis=0).mean(); between=cols.mean(axis=0).std()
    print(f"{frac:6.0%} {a_full:13.1%} {a_one:22.1%} {np.median(seps):22.2f} {within:14.3f} / {between:.3f}")
print("\n(chance: full profile 5.9%, sentiment-anchor 6.2%)")
