import json, numpy as np
from pathlib import Path
DATA = Path.home()/"rosetta_data"/"paper_n250"
C = ["agency","authorization","causation","certainty","credibility","deception","exfiltration","formality",
     "moral_valence","negation","plurality","sarcasm","sentiment","specificity","temporal_order","threat_severity","urgency"]
models=[d.name for d in sorted(DATA.iterdir()) if d.is_dir() and not d.name.endswith(("_it","_Instruct")) and not d.name.startswith("_")
        and all((d/f"caz_{c}.json").exists() for c in C)]
def load(m):
    M=[]
    for c in C:
        ld=json.load(open(DATA/m/f"caz_{c}.json"))["layer_data"]; pk=ld["peak_layer"]
        M.append(np.array(ld["metrics"][pk]["dom_vector"],float))
    M=np.vstack(M); return M/np.linalg.norm(M,axis=1,keepdims=True)
G={m:load(m)@load(m).T for m in models}
print(f"{len(models)} base models with all 17 concepts:"); print("  "+", ".join(models))
ia=C.index("sentiment"); ib=C.index("authorization")
print("\n== ONE ANCHOR (number line): cosine to SENTIMENT, per concept, across models ==")
print(f"{'concept':16s} {'mean':>6s} {'sd':>6s}  {'min':>6s} {'max':>6s}")
rows=[]
for j,c in enumerate(C):
    if j==ia: continue
    v=np.array([G[m][ia,j] for m in models]); rows.append((c,v))
for c,v in sorted(rows,key=lambda t:-t[1].mean()):
    print(f"{c:16s} {v.mean():+.3f} {v.std():.3f}  {v.min():+.3f} {v.max():+.3f}")
within=np.mean([v.std() for c,v in rows]); between=np.std([v.mean() for c,v in rows])
print(f"\nwithin-concept sd across models (mean): {within:.3f}   between-concept sd of means: {between:.3f}   ratio between/within: {between/within:.2f}")

print("\n== TWO ANCHORS (plane): sentiment & authorization — cluster separability of concept identity ==")
# relative coords for each (model, concept): cosines to anchors; leave-one-model-out nearest-centroid classification
def loo_acc(anchor_idx):
    others=[j for j in range(17) if j not in anchor_idx]
    X={m:np.array([[G[m][a,j] for a in anchor_idx] for j in others]) for m in models}
    correct=0; total=0
    for m in models:
        cent=np.mean([X[o] for o in models if o!=m],axis=0)  # centroid per concept
        for k in range(len(others)):
            pred=np.argmin(np.linalg.norm(cent-X[m][k],axis=1)); correct+=(pred==k); total+=1
    return correct/total, len(others)
for anchors,name in [([ia],"sentiment only"),([ia,ib],"sentiment + authorization"),([ia,ib,C.index("negation")],"sentiment + authorization + negation"),
                     (list(range(17)),"all 17 as anchors (full relational profile)")]:
    acc,n=loo_acc(anchors) if len(anchors)<17 else (None,None)
    if acc is None:
        # all-17: profile of concept j = row j of Gram excluding diagonal; classify leave-one-model-out
        X={m:np.array([np.delete(G[m][j],j) for j in range(17)]) for m in models}
        correct=0; total=0
        for m in models:
            cent=np.mean([X[o] for o in models if o!=m],axis=0)
            for k in range(17):
                pred=np.argmin(np.linalg.norm(cent-X[m][k],axis=1)); correct+=(pred==k); total+=1
        acc,n=correct/total,17
    print(f"  {name:48s} leave-one-model-out nearest-centroid accuracy: {acc:5.1%}   (chance {1/n:.1%})")
# null: shuffle concept labels within each model
rng=np.random.default_rng(0); accs=[]
for _ in range(200):
    Gp={m:(lambda P:G[m][np.ix_(P,P)])(rng.permutation(17)) for m in models}
    X={m:np.array([np.delete(Gp[m][j],j) for j in range(17)]) for m in models}
    correct=0;total=0
    for m in models:
        cent=np.mean([X[o] for o in models if o!=m],axis=0)
        for k in range(17):
            pred=np.argmin(np.linalg.norm(cent-X[m][k],axis=1)); correct+=(pred==k); total+=1
    accs.append(correct/total)
print(f"  shuffled-label null for the all-17 profile: {np.mean(accs):.1%} ± {np.std(accs):.1%} (max {np.max(accs):.1%} over 200)")
