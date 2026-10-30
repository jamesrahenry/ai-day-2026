import json, numpy as np
from pathlib import Path
DATA = Path.home()/"rosetta_data"/"paper_n250"
C = ["agency","authorization","causation","certainty","credibility","deception","exfiltration","formality",
     "moral_valence","negation","plurality","sarcasm","sentiment","specificity","temporal_order","threat_severity","urgency"]
models=[d.name for d in sorted(DATA.iterdir()) if d.is_dir() and not d.name.startswith("_") and "_it" not in d.name[-3:] and "Instruct" not in d.name
        and all((d/f"caz_{c}.json").exists() for c in C)]
fam=lambda m: next(f for f,k in [("pythia","pythia"),("gptneo","gpt_neo"),("qwen","Qwen"),("opt","opt"),("gemma","gemma"),("llama","Llama"),("phi","phi"),("mistral","istral"),("gpt2","gpt2"),("falcon","falcon")] if k in m)
V={}; S={}
for m in models:
    V[m]={}; S[m]={}
    for c in C:
        ld=json.load(open(DATA/m/f"caz_{c}.json"))["layer_data"]
        M=np.array([x["dom_vector"] for x in ld["metrics"]],float); M/=np.linalg.norm(M,axis=1,keepdims=True)+1e-12
        V[m][c]=M; S[m][c]=np.array([x["separation_fisher"] for x in ld["metrics"]])
frac=0.75
G={}; L_={}
for m in models:
    L=V[m][C[0]].shape[0]; l=int(round(frac*(L-1))); M=np.vstack([V[m][c][l] for c in C]); G[m]=M@M.T; L_[m]=l
print(f"anchor quality at {frac:.0%} depth, 35 models, leave-one-family-out\n")
print(f"{'anchor':16s} {'LOFO acc':>9s} {'between/within':>15s} {'median sep':>11s} {'mean |cos| to others':>21s}")
rows=[]
for ia,a in enumerate(C):
    X={m:np.array([[G[m][ia,j]] for j in range(17) if j!=ia]) for m in models}
    correct=total=0
    for m in models:
        cent=np.mean([X[o] for o in models if fam(o)!=fam(m)],axis=0)
        for k in range(16): correct+=(np.argmin(np.abs(cent-X[m][k]))==k); total+=1
    cols=np.array([[G[m][ia,j] for j in range(17) if j!=ia] for m in models])
    ratio=cols.mean(axis=0).std()/cols.std(axis=0).mean()
    sep=np.median([S[m][a][L_[m]] for m in models])
    rows.append((a,correct/total,ratio,sep,np.abs(cols.mean(axis=0)).mean()))
for a,acc,ratio,sep,ab in sorted(rows,key=lambda r:-r[1]):
    print(f"{a:16s} {acc:9.1%} {ratio:15.2f} {sep:11.2f} {ab:21.3f}")
