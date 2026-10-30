import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
def rsa(A,B):
    iu=np.triu_indices(17,1); return spearmanr((A@A.T)[iu],(B@B.T)[iu]).statistic
rng=np.random.default_rng(0)
A=load("meta-llama/Llama-3.1-8B"); B=load("mistralai/Mistral-7B-v0.3")
G=rng.standard_normal(B.shape); G/=np.linalg.norm(G,axis=1,keepdims=True)
iu=np.triu_indices(17,1); xa=(B@B.T)[iu]; ya=(A@A.T)[iu]
rho=rsa(A,B); null=np.array([rsa(A,B[rng.permutation(17)]) for _ in range(1000)]); rrand=rsa(A,G)
z=(rho-null.mean())/null.std()
print(f"rho={rho:.3f} null={null.mean():.3f}±{null.std():.3f} max={null.max():.3f} random={rrand:.3f} z={z:.1f}")
BLUE="#1D4E89"; ORANGE="#B3541E"; GREY="#8A94A6"; INK="#15181D"
plt.rcParams.update({"font.size":13,"axes.edgecolor":GREY,"axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,7),gridspec_kw={"width_ratios":[1,1.15]}); fig.patch.set_facecolor("white")
ax1.plot([-1,1],[-1,1],ls="--",color=GREY,lw=1.5,zorder=1)
ax1.scatter(xa,ya,s=70,color=BLUE,alpha=0.85,edgecolors="white",linewidths=0.8,zorder=3)
lab={("causation","temporal_order"):(10,-16),("negation","plurality"):(10,8),("credibility","formality"):(-150,10),("sentiment","moral_valence"):(10,-16),("authorization","exfiltration"):(10,8)}
idx={c:i for i,c in enumerate(C)}
for (a,b),off in lab.items():
    i,j=sorted((idx[a],idx[b])); x=(B@B.T)[i,j]; y=(A@A.T)[i,j]
    ax1.annotate(f"{a.replace('_',' ')} × {b.replace('_',' ')}",(x,y),textcoords="offset points",xytext=off,fontsize=11,color=INK,
                 arrowprops=dict(arrowstyle="-",color=GREY,lw=0.8))
lim=max(np.abs(np.concatenate([xa,ya])))*1.1
ax1.set_xlim(-lim,lim); ax1.set_ylim(-lim,lim); ax1.set_aspect("equal")
ax1.axhline(0,color=GREY,lw=0.5,alpha=0.5); ax1.axvline(0,color=GREY,lw=0.5,alpha=0.5)
ax1.set_xlabel("similarity of the concept pair inside Mistral-7B  (cosine)",fontsize=13)
ax1.set_ylabel("similarity of the same pair inside Llama-3.1-8B  (cosine)",fontsize=13)
ax1.set_title(f"136 concept pairs, no rotation fitted\nSpearman ρ = {rho:.2f}",fontsize=15,fontweight="bold",pad=10)
ax1.text(0.97,0.03,"dashed line = perfect agreement",transform=ax1.transAxes,ha="right",va="bottom",fontsize=11,color=GREY)
for s in ("top","right"): ax1.spines[s].set_visible(False)
ax2.hist(null,bins=40,color=GREY,alpha=0.55,edgecolor="white",label=f"concept labels shuffled, 1,000 times\nmean {null.mean():+.2f}, sd {null.std():.2f}, max {null.max():+.2f}")
ax2.axvline(rrand,color=ORANGE,lw=3,ls="--",label=f"Llama vs 17 random vectors: ρ = {rrand:+.2f}")
ax2.axvline(rho,color=BLUE,lw=4,label=f"true pairing, Llama vs Mistral: ρ = {rho:+.2f}  ({z:.0f} sd above the null)")
ax2.set_xlim(-0.5,1.0); ax2.set_xlabel("Spearman ρ between the two models' concept-pair similarities",fontsize=13); ax2.set_ylabel("count of shuffles",fontsize=13)
ax2.set_title("Can the test fail?  Three things it was asked to tell apart",fontsize=15,fontweight="bold",pad=10)
ax2.legend(loc="upper left",fontsize=12,frameon=False)
for s in ("top","right"): ax2.spines[s].set_visible(False)
fig.suptitle("Do Llama-3.1-8B and Mistral-7B agree on how the 17 concepts relate to each other?",fontsize=16,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.94]); plt.savefig("fig_rsa_null_llama_mistral.png",dpi=170,bbox_inches="tight"); print("saved")
