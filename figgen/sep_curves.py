import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
DATA = Path.home()/"rosetta_data"/"paper_n250"
C = ["agency","authorization","causation","certainty","credibility","deception","exfiltration","formality",
     "moral_valence","negation","plurality","sarcasm","sentiment","specificity","temporal_order","threat_severity","urgency"]
def curves(model):
    out={}
    for c in C:
        d=json.load(open(DATA/model/f"caz_{c}.json"))["layer_data"]
        s=np.array([m["separation_fisher"] for m in d["metrics"]]); out[c]=(s,d["peak_layer"],d["n_layers"])
    return out
for model,label in [("meta_llama_Llama_3.1_8B","Llama-3.1-8B"),("mistralai_Mistral_7B_v0.3","Mistral-7B")]:
    cv=curves(model); print(label, {c:(int(p), round(float(s.max()),2)) for c,(s,p,n) in cv.items()})
BLUE="#1D4E89"; ORANGE="#B3541E"; GREEN="#2E6B4F"; GREY="#8A94A6"; INK="#15181D"
plt.rcParams.update({"font.size":13,"axes.edgecolor":GREY})
fig,axes=plt.subplots(1,2,figsize=(15,5.6),sharey=False); fig.patch.set_facecolor("white")
hl={"credibility":BLUE,"sarcasm":ORANGE,"authorization":GREEN}
for ax,(model,label) in zip(axes,[("meta_llama_Llama_3.1_8B","Llama-3.1-8B"),("mistralai_Mistral_7B_v0.3","Mistral-7B-v0.3")]):
    cv=curves(model)
    for c,(s,p,n) in cv.items():
        x=np.arange(len(s))/(len(s)-1)*100
        if c in hl: continue
        ax.plot(x,s,color=GREY,lw=1.2,alpha=0.45)
    for c,col in hl.items():
        s,p,n=cv[c]; x=np.arange(len(s))/(len(s)-1)*100
        ax.plot(x,s,color=col,lw=3.2,label=c)
        ax.scatter([x[p]],[s[p]],s=160,color=col,edgecolors="white",linewidths=2,zorder=5)
        ax.annotate(f"peak: layer {p} of {len(s)-1}",(x[p],s[p]),textcoords="offset points",xytext=(10,8),fontsize=11,color=col,fontweight="bold")
    ax.set_title(label,fontsize=16,fontweight="bold",pad=10)
    ax.set_xlabel("depth in the network  (% of layers)",fontsize=13); ax.set_ylabel("how well the layer separates the pairs  (Fisher)",fontsize=13)
    for s_ in ("top","right"): ax.spines[s_].set_visible(False)
    ax.grid(axis="y",color="#E3E7EE",lw=0.8)
axes[0].legend(loc="upper left",fontsize=12,frameon=False,title="3 of 17 concepts highlighted; grey = the other 14",title_fontsize=11)
fig.suptitle("The same prompt pairs, read at every layer: separation grows with depth and peaks at a characteristic layer",fontsize=15,fontweight="bold")
plt.tight_layout(rect=[0,0,1,0.92]); plt.savefig("fig_sep_curves.png",dpi=170,bbox_inches="tight"); print("saved")
