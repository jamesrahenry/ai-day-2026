import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
DATA = Path.home()/"rosetta_data"/"paper_n250"
C = ["agency","authorization","causation","certainty","credibility","deception","exfiltration","formality",
     "moral_valence","negation","plurality","sarcasm","sentiment","specificity","temporal_order","threat_severity","urgency"]
def load(m):
    M=[]
    for c in C:
        ld=json.load(open(DATA/m/f"caz_{c}.json"))["layer_data"]; pk=ld["peak_layer"]
        M.append(np.array(ld["metrics"][pk]["dom_vector"],float))
    M=np.vstack(M); return M/np.linalg.norm(M,axis=1,keepdims=True)
A=load("meta_llama_Llama_3.1_8B"); B=load("mistralai_Mistral_7B_v0.3")
GA=A@A.T; GB=B@B.T
print("per-anchor Spearman between the 16 other-concept cosines, Llama vs Mistral:")
res=[]
for i,c in enumerate(C):
    m=np.arange(17)!=i; r=spearmanr(GA[i,m],GB[i,m]).statistic; res.append((c,r))
for c,r in sorted(res,key=lambda t:-t[1]): print(f"  {c:16s} {r:+.2f}")
import sys
anchor = sys.argv[1] if len(sys.argv)>1 else "sentiment"
i=C.index(anchor); m=np.arange(17)!=i
others=[c for c in C if c!=anchor]; a=GA[i,m]; b=GB[i,m]
order=np.argsort(-a); others=[others[k] for k in order]; a=a[order]; b=b[order]
r=spearmanr(a,b).statistic
BLUE="#1D4E89"; ORANGE="#B3541E"; INK="#15181D"; GREY="#8A94A6"
plt.rcParams.update({"font.size":13})
fig,ax=plt.subplots(figsize=(15,6.2)); fig.patch.set_facecolor("white")
x=np.arange(16); w=0.38
ax.bar(x-w/2,a,w,color=BLUE,label="inside Llama-3.1-8B")
ax.bar(x+w/2,b,w,color=ORANGE,label="inside Mistral-7B-v0.3")
ax.axhline(0,color=INK,lw=1)
ax.set_xticks(x); ax.set_xticklabels([o.replace("_"," ") for o in others],rotation=35,ha="right",fontsize=13)
ax.set_ylabel(f"cosine between the {anchor} vector\nand each other concept's vector",fontsize=13)
ax.set_title(f"Ask each model the same question: what is {anchor} similar to?   Agreement between the two answers: Spearman ρ = {r:.2f}",fontsize=15,fontweight="bold",pad=12)
ax.legend(fontsize=13,frameon=False,loc="upper right")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.grid(axis="y",color="#E3E7EE",lw=0.8)
plt.tight_layout(); plt.savefig(f"fig_anchor_{anchor}.png",dpi=170,bbox_inches="tight"); print("saved",anchor,r)
