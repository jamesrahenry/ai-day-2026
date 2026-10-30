import json, numpy as np
from pathlib import Path
DATA = Path.home()/"rosetta_data"/"paper_n250"
C = ["agency","authorization","causation","certainty","credibility","deception","exfiltration","formality",
     "moral_valence","negation","plurality","sarcasm","sentiment","specificity","temporal_order","threat_severity","urgency"]
models=[d.name for d in sorted(DATA.iterdir()) if d.is_dir() and not d.name.startswith("_") and "_it" not in d.name[-3:] and "Instruct" not in d.name
        and all((d/f"caz_{c}.json").exists() for c in C)]
fam=lambda m: ("pythia" if "pythia" in m else "gptneo" if "gpt_neo" in m else "qwen" if "Qwen" in m else "opt" if "opt" in m else "gemma" if "gemma" in m
               else "llama" if "Llama" in m else "phi" if "phi" in m else "mistral" if "istral" in m else "gpt2" if "gpt2" in m else "falcon" if "falcon" in m else "?")
def load(m):
    M=[]
    for c in C:
        ld=json.load(open(DATA/m/f"caz_{c}.json"))["layer_data"]; pk=ld["peak_layer"]
        M.append(np.array(ld["metrics"][pk]["dom_vector"],float))
    M=np.vstack(M); return M/np.linalg.norm(M,axis=1,keepdims=True)
G={m:load(m)@load(m).T for m in models}
fams=sorted(set(fam(m) for m in models)); print(f"{len(models)} base models, {len(fams)} families: {fams}")
def feats(Gm, anchors):
    if anchors is None: return np.array([np.delete(Gm[j],j) for j in range(17)]), list(range(17))
    others=[j for j in range(17) if j not in anchors]; return np.array([[Gm[a,j] for a in anchors] for j in others]), others
def acc(Gd, anchors, holdout):  # holdout: function m -> group key
    X={m:feats(Gd[m],anchors)[0] for m in models}; correct=total=0
    for m in models:
        train=[o for o in models if holdout(o)!=holdout(m)]
        cent=np.mean([X[o] for o in train],axis=0)
        for k in range(X[m].shape[0]):
            correct+=(np.argmin(np.linalg.norm(cent-X[m][k],axis=1))==k); total+=1
    return correct/total
ia,ib,ic=C.index("sentiment"),C.index("authorization"),C.index("negation")
print(f"\n{'anchors':46s} {'leave-one-MODEL-out':>20s} {'leave-one-FAMILY-out':>21s} {'chance':>7s}")
for anchors,name in [([ia],"sentiment"),([ia,ib],"sentiment + authorization"),([ia,ib,ic],"sentiment + authorization + negation"),(None,"all 17 (full relational profile)")]:
    n=17 if anchors is None else 17-len(anchors)
    print(f"{name:46s} {acc(G,anchors,lambda m:m):19.1%} {acc(G,anchors,fam):21.1%} {1/n:7.1%}")
rng=np.random.default_rng(1); null=[]
for _ in range(200):
    Gp={m:(lambda P:G[m][np.ix_(P,P)])(rng.permutation(17)) for m in models}; null.append(acc(Gp,None,fam))
print(f"\nshuffled-label null, full profile, leave-one-family-out: {np.mean(null):.1%} ± {np.std(null):.1%} (max {np.max(null):.1%})")
# per-concept accuracy, full profile, LOFO
X={m:feats(G[m],None)[0] for m in models}; per={c:0 for c in C}
for m in models:
    cent=np.mean([X[o] for o in models if fam(o)!=fam(m)],axis=0)
    for k in range(17): per[C[k]]+=(np.argmin(np.linalg.norm(cent-X[m][k],axis=1))==k)
print("\nper-concept identification rate (leave-one-family-out, full profile):")
for c,v in sorted(per.items(),key=lambda t:-t[1]): print(f"  {c:16s} {v/len(models):5.0%}")
