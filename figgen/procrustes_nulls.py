import json, sys, numpy as np
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
DATA = Path.home()/"rosetta_data"/"paper_n250"
C = ["agency","authorization","causation","certainty","credibility","deception","exfiltration","formality",
     "moral_valence","negation","plurality","sarcasm","sentiment","specificity","temporal_order","threat_severity","urgency"]
def key(m): return m.replace("/","_").replace("-","_")
def load(m):
    M=[]
    for c in C:
        ld=json.load(open(DATA/key(m)/f"caz_{c}.json"))["layer_data"]; pk=ld["peak_layer"]
        M.append(np.array(ld["metrics"][pk]["dom_vector"],float))
    M=np.vstack(M); return M/np.linalg.norm(M,axis=1,keepdims=True)
def insample(A,B):
    R,_=orthogonal_procrustes(B,A); return np.mean(np.sum((B@R)*A,1))
def loco(A,B):
    out=[]
    for i in range(len(C)):
        m=np.arange(len(C))!=i
        R,_=orthogonal_procrustes(B[m],A[m]); out.append(float((B[i]@R)@A[i]))
    return np.array(out)
rng=np.random.default_rng(0)
A=load("meta-llama/Llama-3.1-8B"); B=load("mistralai/Mistral-7B-v0.3")
d=A.shape[1]
print(f"d={d}  n={len(C)}")
print(f"Gram off-diagonal |cos| mean: Llama {np.abs(A@A.T-np.eye(17)).sum()/(17*16):.3f}  Mistral {np.abs(B@B.T-np.eye(17)).sum()/(17*16):.3f}")
print(f"raw same-concept cosine (no rotation):        {np.mean(np.sum(A*B,1)):+.3f}")
print(f"IN-SAMPLE Procrustes, true pairing:          {insample(A,B):.3f}")
perm=[insample(A,B[rng.permutation(17)]) for _ in range(200)]
print(f"IN-SAMPLE Procrustes, permuted labels:       {np.mean(perm):.3f} ± {np.std(perm):.3f}  (200 perms)")
G=rng.standard_normal(B.shape); G/=np.linalg.norm(G,axis=1,keepdims=True)
print(f"IN-SAMPLE Procrustes, Llama vs random Gauss: {insample(A,G):.3f}")
l=loco(A,B)
print(f"LOCO held-out cosine, true pairing:          {l.mean():.3f} ± {l.std():.3f}   min {l.min():.2f} max {l.max():.2f}")
lp=np.array([loco(A,B[rng.permutation(17)]).mean() for _ in range(200)])
print(f"LOCO held-out cosine, permuted labels:       {lp.mean():+.3f} ± {lp.std():.3f}  (200 perms; max {lp.max():.3f})")
lg=loco(A,G); print(f"LOCO held-out cosine, Llama vs random Gauss: {lg.mean():+.3f}")
print("\nper-concept LOCO held-out cosine:")
for c,v in sorted(zip(C,l), key=lambda t:-t[1]): print(f"  {c:16s} {v:+.3f}")
