import json, sys, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pathlib import Path
for _p in (Path.home()/"rosetta_tools", Path.home()/"Source"/"Rosetta_Program"/"rosetta_tools"):  # GPU-host layout first
    if (_p/"rosetta_tools").is_dir(): sys.path.insert(0, str(_p)); break
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
# Drawn at its printed size on the A1 poster (about 524 x 105 mm), so 12 pt here prints as 12 pt.
W,H=20.6,4.15; FS=12; LABEL_FS=11.5
plt.rcParams.update({"axes.axisbelow":True,"font.size":FS,"axes.edgecolor":GREY,"font.family":"DejaVu Sans"})
fig=plt.figure(figsize=(W,H)); fig.patch.set_facecolor("white")
gs=gridspec.GridSpec(1,3,width_ratios=[2.35,1.05,1.0],wspace=0.16,left=0.058,right=0.988,bottom=0.15,top=0.9)
ax=fig.add_subplot(gs[0,0]); axn=fig.add_subplot(gs[0,1],sharey=ax); ax2=fig.add_subplot(gs[0,2])
def ribbons(a, data, thin=True):
    finals=[]
    for c in others:
        col=concept_color(c); Y=data[c]
        if thin:
            for y in Y: a.plot(grid*100,y,color=col,lw=0.4,alpha=0.12)
        m_=np.median(Y,axis=0); q1=np.percentile(Y,25,axis=0); q3=np.percentile(Y,75,axis=0)
        a.fill_between(grid*100,q1,q3,color=col,alpha=0.28,lw=0); a.plot(grid*100,m_,color=col,lw=2.2)
        finals.append((m_[-1],c,col))
    a.axhline(0,color=INK,lw=0.8); a.set_xticks([0,25,50,75,100]); a.grid(axis="y",color="#E3E7EE",lw=0.7)
    for s_ in ("top","right"): a.spines[s_].set_visible(False)
    return finals
YLIM=(-0.95,0.98)
finals=ribbons(ax,cos)
# Space the end labels so none overlap: one label line in data units, from the font size and axes height.
plot_h_in=H*(0.9-0.15); line=LABEL_FS*1.18/72/plot_h_in*(YLIM[1]-YLIM[0])
finals.sort(); ys=[f[0] for f in finals]
for i in range(1,len(ys)):
    if ys[i]-ys[i-1]<line: ys[i]=ys[i-1]+line
top=YLIM[1]-line/2
if ys[-1]>top: ys=[y-(ys[-1]-top) for y in ys]   # shift the stack down if it runs off the top
for i in range(len(ys)-2,-1,-1):                  # then re-space downwards
    if ys[i+1]-ys[i]<line: ys[i]=ys[i+1]-line
for (v,c,col),y in zip(finals,ys):
    ax.annotate(c.replace("_"," "),xy=(100,v),xytext=(104,y),textcoords="data",fontsize=LABEL_FS,color=col,va="center",
                fontweight="bold",arrowprops=dict(arrowstyle="-",color=col,lw=0.7,alpha=0.7,shrinkA=0,shrinkB=0))
ax.set_xlim(0,122); ax.set_ylim(*YLIM)
ax.set_xlabel("depth in the network (% of layers)"); ax.set_ylabel(f"cosine to {A}\n(sign = label polarity)")
ax.set_title("Real labels, 35 models",fontsize=12.5,fontweight="bold",loc="left",pad=6)
ax.set_title("thin line = one model · band = middle half · thick = median",fontsize=FS-1,loc="right",pad=6,color="#3C4450")
ribbons(axn,cosn,thin=False)
axn.set_xlim(0,100); axn.tick_params(labelleft=False); axn.set_xlabel("depth in the network (% of layers)")
axn.set_title("Null: labels shuffled within each model",fontsize=12.5,fontweight="bold",loc="left",pad=6)
axn.text(50,0.8,"no structure survives the shuffle",ha="center",va="top",fontsize=FS,color=INK,style="italic")
ax2.plot(fr*100,np.array(acc_real)*100,color=BLUE,lw=2.6,marker="o",ms=5,label="real labels")
ax2.plot(fr*100,np.array(acc_shuf)*100,color=GREY,lw=2,marker="s",ms=4.5,label="labels shuffled (mean of 20)")
ax2.plot(fr*100,np.array(acc_rand)*100,color=ORANGE,lw=2,marker="^",ms=5,ls="--",label="17 random vectors per model")
ax2.axhline(100/17,color=INK,lw=0.8,ls=":"); ax2.text(99,100/17+3,"chance: 1 in 17",color=INK,fontsize=FS-1,ha="right")
ax2.set_ylim(0,102); ax2.set_xlim(0,100); ax2.set_xticks([0,25,50,75,100]); ax2.set_xlabel("depth in the network (% of layers)")
ax2.set_ylabel("named correctly (%)")
ax2.set_title("Naming concepts in an unseen family",fontsize=12.5,fontweight="bold",loc="left",pad=6)
ax2.legend(loc="center right",fontsize=FS-1,frameon=False)
for s_ in ("top","right"): ax2.spines[s_].set_visible(False)
ax2.grid(axis="y",color="#E3E7EE",lw=0.7)
plt.savefig(f"fig_ribbon_{A}_null.png",dpi=300); print("saved")
