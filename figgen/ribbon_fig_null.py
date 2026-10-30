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
A=sys.argv[1] if len(sys.argv)>1 else "credibility"; ia=C.index(A); others=[c for c in C if c!=A]
V={}
for m in models:
    V[m]={}
    for c in C:
        ld=json.load(open(DATA/m/f"caz_{c}.json"))["layer_data"]
        M=np.array([x["dom_vector"] for x in ld["metrics"]],float); M/=np.linalg.norm(M,axis=1,keepdims=True)+1e-12; V[m][c]=M
grid=np.linspace(0,1,41); rng=np.random.default_rng(0)
def interp(y,L): return np.interp(grid, np.linspace(0,1,L), y)
# real cosines to anchor, models x grid, per concept
cos={c:np.array([interp(np.sum(V[m][A]*V[m][c],axis=1), V[m][A].shape[0]) for m in models]) for c in others}
# shuffled-label null: one independent permutation of the 17 labels per model
perm={m:rng.permutation(17) for m in models}
def cos_null(c):
    out=[]
    for m in models:
        P=perm[m]; a=C[P[ia]]; o=C[P[C.index(c)]]
        out.append(interp(np.sum(V[m][a]*V[m][o],axis=1), V[m][a].shape[0]))
    return np.array(out)
cosn={c:cos_null(c) for c in others}
# identification accuracy by depth: real, shuffled (20 perms), random vectors
def gram(m,frac,Mat=None):
    L=V[m][C[0]].shape[0]; l=int(round(frac*(L-1)))
    M=np.vstack([V[m][c][l] for c in C]) if Mat is None else Mat; return M@M.T
def lofo(Gd):
    X={m:np.array([np.delete(Gd[m][j],j) for j in range(17)]) for m in models}; correct=total=0
    for m in models:
        cent=np.mean([X[o] for o in models if fam(o)!=fam(m)],axis=0)
        for k in range(17): correct+=(np.argmin(np.linalg.norm(cent-X[m][k],axis=1))==k); total+=1
    return correct/total
fr=np.linspace(0,1,11); acc_real=[]; acc_shuf=[]; acc_rand=[]
for f in fr:
    G={m:gram(m,f) for m in models}; acc_real.append(lofo(G))
    acc_shuf.append(np.mean([lofo({m:(lambda P:G[m][np.ix_(P,P)])(rng.permutation(17)) for m in models}) for _ in range(20)]))
    GR={}
    for m in models:
        R=rng.standard_normal((17,V[m][C[0]].shape[1])); R/=np.linalg.norm(R,axis=1,keepdims=True); GR[m]=R@R.T
    acc_rand.append(lofo(GR))
print("real",np.round(acc_real,3)); print("shuf",np.round(acc_shuf,3)); print("rand",np.round(acc_rand,3))

INK="#15181D"; GREY="#8A94A6"; BLUE="#1D4E89"; ORANGE="#B3541E"
plt.rcParams.update({"font.size":16.5,"axes.edgecolor":GREY})
fig=plt.figure(figsize=(30,6.4)); fig.patch.set_facecolor("white")
gs=gridspec.GridSpec(1,3,width_ratios=[2.3,1.1,1.0],wspace=0.2)
ax=fig.add_subplot(gs[0,0]); axn=fig.add_subplot(gs[0,1],sharey=ax); ax2=fig.add_subplot(gs[0,2])
def ribbons(a, data, thin=True):
    finals=[]
    for c in others:
        col=concept_color(c); Y=data[c]
        if thin:
            for y in Y: a.plot(grid*100,y,color=col,lw=0.6,alpha=0.12)
        m_=np.median(Y,axis=0); q1=np.percentile(Y,25,axis=0); q3=np.percentile(Y,75,axis=0)
        a.fill_between(grid*100,q1,q3,color=col,alpha=0.28,lw=0); a.plot(grid*100,m_,color=col,lw=3)
        finals.append((m_[-1],c,col))
    a.axhline(0,color=INK,lw=1); a.set_xticks([0,25,50,75,100]); a.grid(axis="y",color="#E3E7EE",lw=0.8)
    for s_ in ("top","right"): a.spines[s_].set_visible(False)
    return finals
finals=ribbons(ax,cos)
finals.sort(); ys=[f[0] for f in finals]; minsep=0.05
for i in range(1,len(ys)):
    if ys[i]-ys[i-1]<minsep: ys[i]=ys[i-1]+minsep
off=(ys[-1]-finals[-1][0])*0.5
for (v,c,col),y in zip(finals,ys):
    ax.annotate(c.replace("_"," "),xy=(100,v),xytext=(103.5,y-off),textcoords="data",fontsize=14.5,color=col,va="center",
                arrowprops=dict(arrowstyle="-",color=col,lw=0.8,alpha=0.6))
ax.set_xlim(0,124); ax.set_ylim(-0.8,0.95)
ax.set_xlabel("depth in the network  (% of layers)"); ax.set_ylabel(f"cosine to the model's own {A} vector")
ax.set_title(f"Real labels: where each concept sits relative to {A}, 35 models\nthin line = one model · band = middle half · thick = median",fontsize=16,fontweight="bold",loc="left",pad=10)
ribbons(axn,cosn,thin=False)
axn.set_xlim(0,100); axn.tick_params(labelleft=False); axn.set_xlabel("depth in the network  (% of layers)")
axn.set_title("Null: labels shuffled within each model\nsame drawing, bands collapse together",fontsize=16,fontweight="bold",loc="left",pad=10)
axn.text(50,0.85,"no structure survives\nthe shuffle",ha="center",va="top",fontsize=14.5,color=INK,style="italic")
ax2.plot(fr*100,np.array(acc_real)*100,color=BLUE,lw=3.5,marker="o",label="real labels")
ax2.plot(fr*100,np.array(acc_shuf)*100,color=GREY,lw=2.5,marker="s",ls="-",label="labels shuffled (mean of 20)")
ax2.plot(fr*100,np.array(acc_rand)*100,color=ORANGE,lw=2.5,marker="^",ls="--",label="17 random vectors per model")
ax2.axhline(100/17,color=INK,lw=1,ls=":"); ax2.text(99,100/17+2.5,"chance 1 in 17",color=INK,fontsize=13.5,ha="right")
ax2.set_ylim(0,100); ax2.set_xlim(0,100); ax2.set_xlabel("depth in the network  (% of layers)")
ax2.set_ylabel("concepts correctly named in a never-seen family  (%)")
ax2.set_title("Can the angles alone name a concept\nin a never-seen model family?",fontsize=16,fontweight="bold",loc="left",pad=10)
ax2.legend(loc="center right",fontsize=14,frameon=False)
for s_ in ("top","right"): ax2.spines[s_].set_visible(False)
ax2.grid(axis="y",color="#E3E7EE",lw=0.8)
fig.suptitle(f"The same question asked of 35 models at every depth: what is {A} similar to?   And what the answer looks like when it is not allowed to mean anything.",fontsize=18,fontweight="bold",y=1.02)
plt.savefig(f"fig_ribbon_{A}_null.png",dpi=170,bbox_inches="tight"); print("saved")
