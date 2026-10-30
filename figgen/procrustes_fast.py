import json, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
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
# In-sample full-space Procrustes optimum: max_R tr(R^T B^T A) = ||B^T A||_* = sum sqrt(eig((B B^T)(A A^T)))
def insample(A,B):
    ev=np.linalg.eigvals((B@B.T)@(A@A.T)).real; return np.sqrt(np.clip(ev,0,None)).sum()/len(A)
def rsa(A,B):
    iu=np.triu_indices(len(A),1); return spearmanr((A@A.T)[iu],(B@B.T)[iu]).statistic
rng=np.random.default_rng(0)
A=load("meta-llama/Llama-3.1-8B"); B=load("mistralai/Mistral-7B-v0.3"); d=A.shape[1]
G=rng.standard_normal(B.shape); G/=np.linalg.norm(G,axis=1,keepdims=True)
print(f"d={d}, n=17\n")
print("IN-SAMPLE PROCRUSTES mean aligned cosine (what the hero figure reports)")
print(f"  true pairing            {insample(A,B):.3f}")
p=[insample(A,B[rng.permutation(17)]) for _ in range(1000)]
print(f"  shuffled concept labels {np.mean(p):.3f} ± {np.std(p):.3f}")
print(f"  Llama vs random vectors {insample(A,G):.3f}")
print("\nRELATIONAL STRUCTURE (Spearman rho between the two 17x17 inter-concept cosine matrices, 136 pairs)")
print(f"  true pairing            {rsa(A,B):+.3f}")
p=[rsa(A,B[rng.permutation(17)]) for _ in range(1000)]
print(f"  shuffled concept labels {np.mean(p):+.3f} ± {np.std(p):.3f}   (max of 1000: {np.max(p):+.3f})")
print(f"  Llama vs random vectors {rsa(A,G):+.3f}")
