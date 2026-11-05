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
# Drawn at its printed size on the A1 poster (about 224 x 90 mm), so 11 pt here prints as 11 pt.
plt.rcParams.update({"axes.axisbelow":True,"font.size":11,"axes.edgecolor":GREY,"font.family":"DejaVu Sans"})
fig,axes=plt.subplots(1,2,figsize=(8.8,3.45),sharey=True); fig.patch.set_facecolor("white")
hl={"credibility":BLUE,"sarcasm":ORANGE,"authorization":GREEN}
for ax,(model,label) in zip(axes,[("meta_llama_Llama_3.1_8B","Llama-3.1-8B"),("mistralai_Mistral_7B_v0.3","Mistral-7B-v0.3")]):
    cv=curves(model)
    for c,(s,p,n) in cv.items():
        if c in hl: continue
        x=np.arange(len(s))/(len(s)-1)*100
        ax.plot(x,s,color=GREY,lw=0.9,alpha=0.45)
    for c,col in hl.items():
        s,p,n=cv[c]; x=np.arange(len(s))/(len(s)-1)*100
        ax.plot(x,s,color=col,lw=2.4,label=c)
        ax.scatter([x[p]],[s[p]],s=55,color=col,edgecolors="white",linewidths=1.5,zorder=5)
        right=x[p]>70  # keep labels for late peaks inside the axes
        ax.annotate(f"peak: layer {p} of {len(s)-1}",(x[p],s[p]),textcoords="offset points",
                    xytext=(-6,7) if right else (6,7),ha="right" if right else "left",fontsize=9.5,color=col,fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12",fc="white",ec="none",alpha=0.85),zorder=6)
    ax.set_title(label,fontsize=12,fontweight="bold",pad=4)
    ax.set_xlabel("depth in the network (% of layers)"); ax.set_xticks([0,25,50,75,100])
    for s_ in ("top","right"): ax.spines[s_].set_visible(False)
    ax.grid(axis="y",color="#E3E7EE",lw=0.7)
axes[0].set_ylabel("separation of the pairs (Fisher)")
h,l=axes[0].get_legend_handles_labels()
h.append(plt.Line2D([],[],color=GREY,lw=1.2,alpha=0.7)); l.append("the other 14 concepts")
fig.legend(h,l,loc="upper center",ncol=4,frameon=False,fontsize=10.5,bbox_to_anchor=(0.5,1.0),handlelength=1.6,columnspacing=1.6)
plt.tight_layout(rect=[0,0,1,0.9],w_pad=1.5); plt.savefig("fig_sep_curves.png",dpi=300,bbox_inches="tight",pad_inches=0.03); print("saved")
