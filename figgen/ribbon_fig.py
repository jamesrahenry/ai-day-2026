import json, sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
sys.path.insert(0, str(Path.home()/"Source"/"Rosetta_Program"))
from rosetta_tools.viz_style import concept_color
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
import sys as _s
grid=np.linspace(0,1,41); A=_s.argv[1] if len(_s.argv)>1 else "sentiment"; others=[c for c in C if c!=A]
def interp(y,L): return np.interp(grid, np.linspace(0,1,L), y)
cos={c:np.array([interp(np.sum(V[m][A]*V[m][c],axis=1), V[m][A].shape[0]) for m in models]) for c in others}   # models x grid
sep=np.array([interp(S[m][A], len(S[m][A])) for m in models])
# identification accuracy by depth (full profile, leave-one-family-out)
def acc_at(frac):
    X={}
    for m in models:
        L=V[m][A].shape[0]; l=int(round(frac*(L-1))); M=np.vstack([V[m][c][l] for c in C]); G=M@M.T
        X[m]=np.array([np.delete(G[j],j) for j in range(17)])
    correct=total=0
    for m in models:
        cent=np.mean([X[o] for o in models if fam(o)!=fam(m)],axis=0)
        for k in range(17): correct+=(np.argmin(np.linalg.norm(cent-X[m][k],axis=1))==k); total+=1
    return correct/total
def acc_one(frac):
    X={}
    for m in models:
        L=V[m][A].shape[0]; l=int(round(frac*(L-1))); M=np.vstack([V[m][c][l] for c in C]); G=M@M.T; ia=C.index(A)
        X[m]=np.array([[G[ia,j]] for j in range(17) if j!=ia])
    correct=total=0
    for m in models:
        cent=np.mean([X[o] for o in models if fam(o)!=fam(m)],axis=0)
        for k in range(16): correct+=(np.argmin(np.linalg.norm(cent-X[m][k],axis=1))==k); total+=1
    return correct/total
fr=np.linspace(0,1,11); acc_full=[acc_at(f) for f in fr]; acc_1=[acc_one(f) for f in fr]
print("acc full:",np.round(acc_full,3)); print("acc one:",np.round(acc_1,3))

INK="#15181D"; GREY="#8A94A6"; BLUE="#1D4E89"; ORANGE="#B3541E"
plt.rcParams.update({"font.size":13,"axes.edgecolor":GREY})
fig=plt.figure(figsize=(17,6.4)); fig.patch.set_facecolor("white")
gs=gridspec.GridSpec(1,2,width_ratios=[2.3,1],wspace=0.2)
ax=fig.add_subplot(gs[0,0]); ax2=fig.add_subplot(gs[0,1])
# ribbons
finals=[]
for c in others:
    col=concept_color(c); Y=cos[c]
    for y in Y: ax.plot(grid*100,y,color=col,lw=0.6,alpha=0.12)
    m_=np.median(Y,axis=0); q1=np.percentile(Y,25,axis=0); q3=np.percentile(Y,75,axis=0)
    ax.fill_between(grid*100,q1,q3,color=col,alpha=0.28,lw=0)
    ax.plot(grid*100,m_,color=col,lw=3)
    finals.append((m_[-1],c,col))
# stacked labels at right edge
finals.sort(); ys=[f[0] for f in finals]; minsep=0.045
for i in range(1,len(ys)):
    if ys[i]-ys[i-1]<minsep: ys[i]=ys[i-1]+minsep
shift=(ys[-1]-finals[-1][0])/2 if ys[-1]>finals[-1][0] else 0
for (v,c,col),y in zip(finals,ys):
    ax.annotate(c.replace("_"," "),xy=(100,v),xytext=(103.5,y-shift*0.3),textcoords="data",fontsize=12,color=col,va="center",
                arrowprops=dict(arrowstyle="-",color=col,lw=0.8,alpha=0.6))
ax.axhline(0,color=INK,lw=1); ax.set_xlim(0,122); ax.set_ylim(-0.75,0.95)
ax.set_xticks([0,25,50,75,100]); ax.set_xlabel("depth in the network  (% of layers)",fontsize=13)
ax.set_ylabel(f"cosine to the model's own {A.replace('_',' ')} vector at that layer",fontsize=13)
ax.set_title(f"Where every other concept sits relative to {A.replace('_',' ')}, in 35 models\nthin line = one model, band = middle half of models, thick = median",fontsize=13.5,fontweight="bold",loc="left",pad=10)
for s_ in ("top","right"): ax.spines[s_].set_visible(False)
ax.grid(axis="y",color="#E3E7EE",lw=0.8)
# accuracy panel
ax2.plot(fr*100,np.array(acc_full)*100,color=BLUE,lw=3.5,marker="o",label="all angles to the other 16 concepts")
ax2.plot(fr*100,np.array(acc_1)*100,color=ORANGE,lw=3,marker="o",label=f"angle to {A.replace('_',' ')} only")
ax2.axhline(100/17,color=GREY,lw=2,ls="--"); ax2.text(2,100/17+2,"chance (1 in 17)",color=GREY,fontsize=11)
ax2.set_ylim(0,100); ax2.set_xlim(0,100); ax2.set_xlabel("depth in the network  (% of layers)",fontsize=13)
ax2.set_ylabel("concepts correctly named  (%)   leave one of 10 families out",fontsize=12)
ax2.set_title("Can the angles alone name a concept\nin a model family never seen before?",fontsize=13.5,fontweight="bold",pad=10)
ax2.legend(loc="center right",fontsize=11.5,frameon=False)
for s_ in ("top","right"): ax2.spines[s_].set_visible(False)
ax2.grid(axis="y",color="#E3E7EE",lw=0.8)
fig.suptitle(f"The same question asked of 35 models at every depth: what is {A.replace('_',' ')} similar to?",fontsize=16,fontweight="bold",y=1.0)
plt.savefig(f"fig_ribbon_{A}.png",dpi=170,bbox_inches="tight"); print("saved")
