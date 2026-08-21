import { useState, useEffect, useRef, useCallback } from "react";
import {
  AreaChart, Area, BarChart, Bar, RadarChart, Radar,
  PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer
} from "recharts";

// ══════════════════════════════════════════════════════════
// CONSTANTS
// ══════════════════════════════════════════════════════════
const GESTURES=["FIST","OPEN_HAND","WRIST_UP","WRIST_DOWN","DOUBLE_FLEX","RELAX"];
const G_ICON ={FIST:"✊",OPEN_HAND:"🖐",WRIST_UP:"☝️",WRIST_DOWN:"👇",DOUBLE_FLEX:"💪",RELAX:"✋",UNKNOWN:"❓"};
const G_LABEL={FIST:"Fist",OPEN_HAND:"Open Hand",WRIST_UP:"Wrist Up",WRIST_DOWN:"Wrist Down",DOUBLE_FLEX:"Double Flex",RELAX:"Relax",UNKNOWN:"Unknown"};
const G_COLOR={FIST:"#dc2626",OPEN_HAND:"#16a34a",WRIST_UP:"#2563eb",WRIST_DOWN:"#d97706",DOUBLE_FLEX:"#9333ea",RELAX:"#475569",UNKNOWN:"#94a3b8"};
const G_BG   ={FIST:"#fef2f2",OPEN_HAND:"#f0fdf4",WRIST_UP:"#eff6ff",WRIST_DOWN:"#fefce8",DOUBLE_FLEX:"#faf5ff",RELAX:"#f8fafc",UNKNOWN:"#f8fafc"};
const CAL_G  =["RELAX","FIST","OPEN_HAND","WRIST_UP","WRIST_DOWN","DOUBLE_FLEX"];
const CAL_INST={
  RELAX:"Rest your forearm flat on a surface. Stay completely relaxed.",
  FIST:"Close your hand into a firm fist and hold the contraction steady.",
  OPEN_HAND:"Extend and spread all five fingers as wide as possible.",
  WRIST_UP:"Bend your wrist upward (dorsiflexion). Hold steady.",
  WRIST_DOWN:"Bend your wrist downward (palmar flexion). Hold steady.",
  DOUBLE_FLEX:"Make a firm fist AND flex your forearm hard simultaneously.",
};
const CAL_EXP_RMS={RELAX:0.040,FIST:0.520,OPEN_HAND:0.310,WRIST_UP:0.420,WRIST_DOWN:0.370,DOUBLE_FLEX:0.670};
const DEFAULT_GMAP={
  light:{on:"FIST",off:"OPEN_HAND"},  fan:{on:"WRIST_UP",off:"WRIST_DOWN"},
  door:{on:"WRIST_DOWN",off:"RELAX"}, motor:{on:"DOUBLE_FLEX",off:"OPEN_HAND"},
  tv:{on:"OPEN_HAND",off:"RELAX"},    ac:{on:"DOUBLE_FLEX",off:"WRIST_UP"},
};
const SR=500, WIN=256, DISP=200;
const REC_DUR=2000, REST_DUR=1500, READY_SEC=3;
const CONF_THRESHOLD=0.52;    // rejection threshold
const SMOOTH_WIN=5;           // temporal smoothing window

// Feature names (15 total)
const FEAT_NAMES=["MAV","MMAV","RMS","VAR","STD","WL","AAC","DASDV","ZC","SSC","IEMG","HjActivity","HjMobility","HjComplexity","MYOP"];

// ══════════════════════════════════════════════════════════
// THEME
// ══════════════════════════════════════════════════════════
const C={
  bg:"#f1f5f9",surf:"#ffffff",input:"#f8fafc",
  border:"#e2e8f0",borderA:"#93c5fd",
  blue:"#2563eb",blueL:"#eff6ff",
  purple:"#7c3aed",purpleL:"#f5f3ff",
  green:"#16a34a",greenL:"#f0fdf4",
  red:"#dc2626",redL:"#fef2f2",
  amber:"#d97706",amberL:"#fefce8",
  t:"#0f172a",t2:"#1e293b",t3:"#475569",t4:"#94a3b8",
};
const card=(x={})=>({background:C.surf,border:`1px solid ${C.border}`,borderRadius:14,padding:18,boxShadow:"0 1px 4px rgba(0,0,0,.07)",...x});
const badge=(color,bg)=>({display:"inline-flex",alignItems:"center",gap:4,background:bg,color,borderRadius:20,padding:"3px 10px",fontSize:11,fontWeight:700});
const btn=(v="primary",x={})=>{
  const s={primary:{background:C.blue,color:"#fff",border:"none"},
    secondary:{background:C.input,color:C.t2,border:`1px solid ${C.border}`},
    success:{background:C.greenL,color:C.green,border:"1px solid #bbf7d0"},
    danger:{background:C.redL,color:C.red,border:"1px solid #fecaca"},
    ghost:{background:"transparent",color:C.t3,border:`1px solid ${C.border}`}};
  return{...s[v],borderRadius:9,padding:"8px 18px",cursor:"pointer",fontWeight:600,fontSize:13,fontFamily:"inherit",transition:"all .15s",...x};
};

// ══════════════════════════════════════════════════════════
// SIGNAL ENGINE
// ══════════════════════════════════════════════════════════
const GP={FIST:{a:0.82,dc:0.022},OPEN_HAND:{a:0.50,dc:0.011},WRIST_UP:{a:0.67,dc:0.016},
          WRIST_DOWN:{a:0.60,dc:0.013},DOUBLE_FLEX:{a:1.02,dc:0.031},RELAX:{a:0.04,dc:0.0}};
let _gn=null;
const gauss=()=>{if(_gn!==null){const v=_gn;_gn=null;return v;}let u,v,s;do{u=2*Math.random()-1;v=2*Math.random()-1;s=u*u+v*v;}while(s>=1||!s);const m=Math.sqrt(-2*Math.log(s)/s);_gn=v*m;return u*m;};

const emgSample=(g,sc,t,hz=120)=>{
  const p=GP[g]??GP.RELAX;
  const am=p.a*(1+0.14*Math.sin(2*Math.PI*1.8*t)+0.06*Math.sin(2*Math.PI*0.4*t));
  return am*gauss()*sc + p.a*sc*0.07*Math.sin(2*Math.PI*hz*t) + 0.016*gauss() + p.dc;
};

const genWin=(g,sc=1,profile={ampF:1,hzF:1})=>
  Array.from({length:WIN},(_,i)=>emgSample(g,sc*profile.ampF,i/SR,120*profile.hzF));

// ══════════════════════════════════════════════════════════
// FEATURE EXTRACTION  — 15 features
// ══════════════════════════════════════════════════════════
function extractFeatures(sig){
  const n=sig.length;
  const mu=sig.reduce((a,b)=>a+b,0)/n;
  const std=Math.sqrt(sig.reduce((a,b)=>a+(b-mu)**2,0)/n);

  // Amplitude features
  const mav =sig.reduce((a,b)=>a+Math.abs(b),0)/n;
  const mmav=sig.reduce((s,x,i)=>s+(i>=n/4&&i<3*n/4?1:0.5)*Math.abs(x),0)/n;
  const rms =Math.sqrt(sig.reduce((a,b)=>a+b*b,0)/n);
  const va  =std*std;
  const iemg=sig.reduce((a,b)=>a+Math.abs(b),0);

  // Morphology
  let wl=0,aac=0,dasdvSum=0;
  const diff1=new Array(n-1), diff2=new Array(n-2);
  for(let i=1;i<n;i++){
    const d=sig[i]-sig[i-1];
    diff1[i-1]=d;
    wl+=Math.abs(d); aac+=Math.abs(d); dasdvSum+=d*d;
  }
  aac/=(n-1);
  const dasdv=Math.sqrt(dasdvSum/(n-1));
  for(let i=0;i<n-2;i++) diff2[i]=diff1[i+1]-diff1[i];

  // Frequency features
  let zc=0,ssc=0;
  for(let i=1;i<n;i++) if((sig[i]>0.01&&sig[i-1]<-0.01)||(sig[i]<-0.01&&sig[i-1]>0.01)) zc++;
  for(let i=1;i<n-1;i++){
    const d1=diff1[i-1]||0, d2=diff1[i]||0;
    if(Math.abs(d1-d2)>0.003&&((d1>0&&d2<0)||(d1<0&&d2>0))) ssc++;
  }

  // Hjorth parameters
  const activity  =va;
  const varD1     =diff1.reduce((a,b)=>a+b*b,0)/(n-1);
  const mobility  =Math.sqrt(varD1/(va+1e-12));
  const varD2     =diff2.reduce((a,b)=>a+b*b,0)/(n-2);
  const complexity=varD1>0?Math.sqrt(varD2/varD1)/(mobility+1e-12):0;

  // Myopulse
  const mythresh=3*std;
  const myop=sig.filter(x=>Math.abs(x)>mythresh).length/n;

  return[mav,mmav,rms,va,std,wl,aac,dasdv,zc,ssc,iemg,activity,mobility,complexity,myop];
}

// Z-score standardisation
function computeScaler(X){
  const n=X.length, nF=X[0].length;
  const mean=new Array(nF).fill(0);
  const std2=new Array(nF).fill(0);
  X.forEach(x=>x.forEach((v,i)=>mean[i]+=v/n));
  X.forEach(x=>x.forEach((v,i)=>std2[i]+=(v-mean[i])**2/n));
  return{mean, std:std2.map(v=>Math.sqrt(v)+1e-9)};
}
function applyScaler(x,sc){return x.map((v,i)=>(v-sc.mean[i])/sc.std[i]);}

// Calibration: per-amplitude feature normalization
function applyCalibration(x,calScale){
  const v=[...x];
  [0,1,2,5,6,7,10].forEach(i=>v[i]/=calScale);
  [3,11].forEach(i=>v[i]/=calScale*calScale);
  v[4]/=calScale;
  return v;
}

// ══════════════════════════════════════════════════════════
// DECISION TREE  (lightweight for browser)
// ══════════════════════════════════════════════════════════
class FastDT{
  constructor(maxDepth=5,nThresh=10,maxFeats=4){this.md=maxDepth;this.nt=nThresh;this.mf=maxFeats;}
  _maj(y){const c={};y.forEach(v=>c[v]=(c[v]||0)+1);return Object.keys(c).reduce((a,b)=>c[a]>c[b]?a:b);}
  _gini(y){const n=y.length;if(!n)return 0;const c={};y.forEach(v=>c[v]=(c[v]||0)+1);return 1-Object.values(c).reduce((s,v)=>s+(v/n)**2,0);}
  _build(X,y,d){
    if(d>=this.md||y.length<=3||new Set(y).size===1)return{leaf:1,cls:this._maj(y)};
    const nF=X[0].length;
    const feats=[...Array(nF).keys()].sort(()=>Math.random()-.5).slice(0,this.mf);
    let bG=-1,bF=null,bT=null;
    const bg=this._gini(y);
    for(const fi of feats){
      const vals=[...new Set(X.map(x=>x[fi]))].sort((a,b)=>a-b);
      const step=Math.max(1,Math.floor(vals.length/this.nt));
      for(let ti=0;ti<vals.length-1;ti+=step){
        const t=(vals[ti]+vals[ti+1])/2;
        const lY=y.filter((_,i)=>X[i][fi]<=t);
        const rY=y.filter((_,i)=>X[i][fi]>t);
        if(!lY.length||!rY.length)continue;
        const g=bg-(lY.length/y.length)*this._gini(lY)-(rY.length/y.length)*this._gini(rY);
        if(g>bG){bG=g;bF=fi;bT=t;}
      }
    }
    if(bF===null)return{leaf:1,cls:this._maj(y)};
    const lI=y.map((_,i)=>X[i][bF]<=bT?i:-1).filter(i=>i>=0);
    const rI=y.map((_,i)=>X[i][bF]>bT?i:-1).filter(i=>i>=0);
    return{leaf:0,f:bF,t:bT,
      l:this._build(lI.map(i=>X[i]),lI.map(i=>y[i]),d+1),
      r:this._build(rI.map(i=>X[i]),rI.map(i=>y[i]),d+1)};
  }
  fit(X,y){this.cls=[...new Set(y)];this.tree=this._build(X,y,0);return this;}
  _pred(node,x){return node.leaf?node.cls:x[node.f]<=node.t?this._pred(node.l,x):this._pred(node.r,x);}
  predict(x){return this._pred(this.tree,x);}
}

// ══════════════════════════════════════════════════════════
// RANDOM FOREST  (30 trees, depth 5, sqrt features)
// ══════════════════════════════════════════════════════════
class RandomForest{
  constructor(nTrees=30,maxDepth=5){this.nT=nTrees;this.mD=maxDepth;}
  fit(X,y){
    this.cls=[...new Set(y)];
    const mf=Math.round(Math.sqrt(X[0].length));
    this.trees=[];
    for(let t=0;t<this.nT;t++){
      const idx=Array.from({length:X.length},()=>Math.floor(Math.random()*X.length));
      new FastDT(this.mD,10,mf).fit(idx.map(i=>X[i]),idx.map(i=>y[i]));
      this.trees.push(new FastDT(this.mD,10,mf).fit(idx.map(i=>X[i]),idx.map(i=>y[i])));
    }
    return this;
  }
  proba(x){
    const v={};this.cls.forEach(c=>v[c]=0);
    this.trees.forEach(t=>{const p=t.predict(x);v[p]++;});
    const tot=this.trees.length;
    return Object.fromEntries(this.cls.map(c=>[c,v[c]/tot]));
  }
  predict(x){const p=this.proba(x);return this.cls.reduce((a,b)=>p[a]>p[b]?a:b);}
  score(X,y){return X.filter((x,i)=>this.predict(x)===y[i]).length/X.length;}
}

// ══════════════════════════════════════════════════════════
// GAUSSIAN NAIVE BAYES
// ══════════════════════════════════════════════════════════
class GNB{
  fit(X,y){
    this.cls=[...new Set(y)];const n=X.length;
    this.pr={};this.mu={};this.va={};
    for(const c of this.cls){
      const Xc=X.filter((_,i)=>y[i]===c);
      this.pr[c]=Xc.length/n;
      this.mu[c]=X[0].map((_,f)=>Xc.reduce((s,x)=>s+x[f],0)/Xc.length);
      this.va[c]=X[0].map((_,f)=>{const m=this.mu[c][f];return Math.max(1e-9,Xc.reduce((s,x)=>s+(x[f]-m)**2,0)/Xc.length);});
    }
    return this;
  }
  _ll(x,c){let l=Math.log(this.pr[c]);for(let f=0;f<x.length;f++){const m=this.mu[c][f],v=this.va[c][f];l-=.5*(Math.log(2*Math.PI*v)+(x[f]-m)**2/v);}return l;}
  proba(x){
    const ls=Object.fromEntries(this.cls.map(c=>[c,this._ll(x,c)]));
    const mx=Math.max(...Object.values(ls));let sm=0;
    const ex=Object.fromEntries(this.cls.map(c=>{const e=Math.exp(ls[c]-mx);sm+=e;return[c,e];}));
    return Object.fromEntries(this.cls.map(c=>[c,ex[c]/sm]));
  }
  predict(x){const p=this.proba(x);return this.cls.reduce((a,b)=>p[a]>p[b]?a:b);}
  score(X,y){return X.filter((x,i)=>this.predict(x)===y[i]).length/X.length;}
}

// ══════════════════════════════════════════════════════════
// WEIGHTED ENSEMBLE  (GNB + RF)
// ══════════════════════════════════════════════════════════
class Ensemble{
  constructor(models,weights){this.models=models;this.weights=weights;}
  proba(x){
    const combined={};
    this.models.forEach((m,mi)=>{
      const p=m.proba(x);
      Object.entries(p).forEach(([c,prob])=>{combined[c]=(combined[c]||0)+prob*this.weights[mi];});
    });
    const total=Object.values(combined).reduce((s,v)=>s+v,0);
    return Object.fromEntries(Object.entries(combined).map(([k,v])=>[k,v/total]));
  }
  predict(x){const p=this.proba(x);return Object.keys(p).reduce((a,b)=>p[a]>p[b]?a:b);}
  score(X,y){return X.filter((x,i)=>this.predict(x)===y[i]).length/X.length;}
}

// ══════════════════════════════════════════════════════════
// TEMPORAL SMOOTHER  (rolling avg of last N probability vectors)
// ══════════════════════════════════════════════════════════
class TemporalSmoother{
  constructor(windowSize=5){this.w=windowSize;this.buf=[];}
  update(proba){
    this.buf.push(proba);
    if(this.buf.length>this.w)this.buf.shift();
    const avg={};
    Object.keys(proba).forEach(c=>{avg[c]=this.buf.reduce((s,p)=>s+(p[c]||0),0)/this.buf.length;});
    const g=Object.keys(avg).reduce((a,b)=>avg[a]>avg[b]?a:b);
    return{gesture:g,confidence:avg[g],proba:avg};
  }
  reset(){this.buf=[];}
}

// ══════════════════════════════════════════════════════════
// MODEL TRAINING  (runs once on load)
// ══════════════════════════════════════════════════════════
let _M=null,_meta=null;

function getModel(){
  if(_M)return{model:_M,meta:_meta,smoother:new TemporalSmoother(SMOOTH_WIN)};

  // Diverse user profiles for training
  const profiles=[];
  for(let i=0;i<15;i++) profiles.push({
    ampF:0.55+i*0.065,                 // 0.55→1.5 amplitude
    hzF :0.85+(i%5)*0.075,             // 0.85→1.15 frequency variation
  });

  const X=[],y=[];
  for(const g of GESTURES){
    for(const prof of profiles){
      for(let tr=0;tr<40;tr++){
        const sig=genWin(g,1.0,prof);
        X.push(extractFeatures(sig));
        y.push(g);
      }
    }
  }

  // Shuffle
  for(let i=X.length-1;i>0;i--){
    const j=Math.floor(Math.random()*(i+1));
    [X[i],X[j]]=[X[j],X[i]];[y[i],y[j]]=[y[j],y[i]];
  }

  // Z-score scaler from training data
  const scaler=computeScaler(X);
  const Xn=X.map(x=>applyScaler(x,scaler));

  // Train/test split
  const sp=Math.floor(Xn.length*0.8);
  const Xt=Xn.slice(0,sp),yt=y.slice(0,sp);
  const Xe=Xn.slice(sp), ye=y.slice(sp);

  // Train GNB
  const gnb=new GNB().fit(Xt,yt);
  const gnbAcc=gnb.score(Xe,ye);

  // Train RF
  const rf=new RandomForest(30,5).fit(Xt,yt);
  const rfAcc=rf.score(Xe,ye);

  // Weighted ensemble (proportional to test accuracy)
  const total=gnbAcc+rfAcc;
  const ens=new Ensemble([gnb,rf],[gnbAcc/total,rfAcc/total]);
  const ensAcc=ens.score(Xe,ye);

  // Confusion matrix
  const cm={};
  GESTURES.forEach(g=>{cm[g]={};GESTURES.forEach(g2=>cm[g][g2]=0);});
  Xe.forEach((x,i)=>cm[ye[i]][ens.predict(x)]++);

  // Feature importance: variance of feature across class means (GNB proxy)
  const classMeans=GESTURES.map(g=>gnb.mu[g]||new Array(15).fill(0));
  const globalMean=classMeans[0].map((_,f)=>classMeans.reduce((s,m)=>s+m[f],0)/classMeans.length);
  const featureImportance=FEAT_NAMES.map((name,f)=>{
    const variance=classMeans.reduce((s,m)=>s+(m[f]-globalMean[f])**2,0)/GESTURES.length;
    return{name,importance:variance};
  }).sort((a,b)=>b.importance-a.importance);
  const totalVar=featureImportance.reduce((s,f)=>s+f.importance,0);
  featureImportance.forEach(f=>f.importance=+(f.importance/(totalVar+1e-9)).toFixed(4));

  _M=ens;
  _meta={
    gnbAcc:+gnbAcc.toFixed(4), rfAcc:+rfAcc.toFixed(4), ensAcc:+ensAcc.toFixed(4),
    trAcc:+ens.score(Xt,yt).toFixed(4), teAcc:+ensAcc.toFixed(4),
    nTr:Xt.length, nTe:Xe.length,
    nFeatures:15, nTrees:30, smoothWin:SMOOTH_WIN,
    scaler, cm, featureImportance,
  };

  return{model:_M,meta:_meta,smoother:new TemporalSmoother(SMOOTH_WIN)};
}

// ══════════════════════════════════════════════════════════
// UI HELPERS
// ══════════════════════════════════════════════════════════
const SL=({children})=>(<div style={{fontSize:10,fontWeight:700,letterSpacing:"0.1em",textTransform:"uppercase",color:C.t4,marginBottom:10}}>{children}</div>);
const Chip=({label,color,bg})=>(<span style={{...badge(color,bg)}}>{label}</span>);
const GesturePill=({g,selected,onClick})=>(
  <button onClick={onClick} style={{background:selected?G_COLOR[g]:G_BG[g],color:selected?"#fff":G_COLOR[g],
    border:`2px solid ${selected?G_COLOR[g]:"transparent"}`,borderRadius:30,padding:"5px 12px",
    cursor:"pointer",fontSize:11,fontWeight:700,transition:"all .15s",fontFamily:"inherit",
    whiteSpace:"nowrap",boxShadow:selected?`0 2px 8px ${G_COLOR[g]}44`:"none"}}>
    {G_ICON[g]} {G_LABEL[g]}
  </button>
);
const NoPill=({selected,onClick})=>(
  <button onClick={onClick} style={{background:selected?C.t3:C.input,color:selected?"#fff":C.t4,
    border:`2px solid ${selected?"transparent":C.border}`,borderRadius:30,padding:"5px 12px",
    cursor:"pointer",fontSize:11,fontWeight:700,transition:"all .15s",fontFamily:"inherit"}}>— None</button>
);
const RepDots=({total,current,phase,color})=>(
  <div style={{display:"flex",gap:6,justifyContent:"center",flexWrap:"wrap"}}>
    {Array.from({length:total},(_,i)=>{
      const done=i<current-1||(i===current-1&&["rest","review","done","complete"].includes(phase));
      const active=i===current-1&&["get_ready","recording"].includes(phase);
      return(<div key={i} style={{width:30,height:30,borderRadius:"50%",display:"flex",alignItems:"center",
        justifyContent:"center",fontSize:12,fontWeight:700,transition:"all .3s",
        background:done?color:active?`${color}18`:"#f1f5f9",
        color:done?"#fff":active?color:C.t4,
        border:`2px solid ${done?color:active?color:C.border}`,
        boxShadow:active?`0 0 0 4px ${color}18`:"none"}}>
        {done?"✓":i+1}
      </div>);
    })}
  </div>
);
const IntensityBar=({value,color=C.blue,label=""})=>(
  <div style={{marginTop:4}}>
    {label&&<div style={{display:"flex",justifyContent:"space-between",fontSize:10,color:C.t4,marginBottom:3}}>
      <span>{label}</span><span style={{fontWeight:700,color}}>{value}%</span>
    </div>}
    <div style={{background:C.input,borderRadius:4,height:9,border:`1px solid ${C.border}`,overflow:"hidden"}}>
      <div style={{width:`${value}%`,height:"100%",background:`linear-gradient(90deg,${color}88,${color})`,borderRadius:4,transition:"width .2s"}}/>
    </div>
  </div>
);

// Storage fallback helpers (works in local browser localStorage and sandbox)
const storageGet = async (key) => {
  try {
    if (typeof window !== "undefined" && window.storage?.get) {
      const r = await window.storage.get(key);
      return r ? r.value : null;
    }
    return typeof localStorage !== "undefined" ? localStorage.getItem(key) : null;
  } catch(e) { return null; }
};

const storageSet = async (key, val) => {
  try {
    if (typeof window !== "undefined" && window.storage?.set) {
      await window.storage.set(key, val);
    } else if (typeof localStorage !== "undefined") {
      localStorage.setItem(key, val);
    }
  } catch(e) {}
};

// ══════════════════════════════════════════════════════════
// MAIN COMPONENT
// ══════════════════════════════════════════════════════════
export default function EMGDashboard(){
  const [ready,    setReady   ]=useState(false);
  const [loading,  setLoading ]=useState("Initialising…");
  const [meta,     setMeta    ]=useState(null);
  const [tab,      setTab     ]=useState("monitor");
  const [live,     setLive    ]=useState(false);
  const [simG,     setSimG    ]=useState("RELAX");
  const [uScale,   setUScale  ]=useState(1.0);
  const [sigData,  setSigData ]=useState([]);
  const [pred,     setPred    ]=useState({g:"RELAX",conf:1,proba:{},intensity:0,snr:0,smoothed:false});
  const [rawFeat,  setRawFeat ]=useState(null);
  const [devs,setDevs]=useState({
    light:{lbl:"Smart Light",ic:"💡",on:false,intensity:100},
    fan:{lbl:"Ceiling Fan",ic:"🌀",on:false,intensity:100},
    door:{lbl:"Auto Door",ic:"🚪",on:false,intensity:100},
    motor:{lbl:"DC Motor",ic:"⚙️",on:false,intensity:100},
    tv:{lbl:"Smart TV",ic:"📺",on:false,intensity:100},
    ac:{lbl:"Air Cond.",ic:"❄️",on:false,intensity:100},
  });
  const [cmdLog,setCmdLog]=useState([]);
  const [gestureMap,setGestureMap]=useState(DEFAULT_GMAP);
  const [profiles,setProfiles]=useState({});
  const [activeProfile,setActiveProfile]=useState(null);
  const [profileName,setProfileName]=useState("Profile 1");
  const [sessionLog,setSessionLog]=useState([]);
  const [sessionTime,setSessionTime]=useState(0);
  const [fatigueWarn,setFatigueWarn]=useState(false);
  // Hardware Connection State
  const [hwMode, setHwMode] = useState(false);
  const [hwStatus, setHwStatus] = useState("Offline (Simulated)");
  const wsRef = useRef(null);

  // Calibration
  const [calPhase,setCalPhase]=useState("idle");
  const [calStep,setCalStep]=useState(0);
  const [calTotalReps,setCalTotalReps]=useState(5);
  const [calRep,setCalRep]=useState(1);
  const [calReadyCnt,setCalReadyCnt]=useState(READY_SEC);
  const [calProg,setCalProg]=useState(0);
  const [calRepData,setCalRepData]=useState([]);
  const [calAllStats,setCalAllStats]=useState({});
  const [calResult,setCalResult]=useState(null);

  // Refs
  const modelRef=useRef(null);
  const smootherRef=useRef(null);
  const scalerRef=useRef(null);
  const bufRef=useRef([]);
  const tick=useRef(0);
  const simGRef=useRef("RELAX");
  const uScaleRef=useRef(1.0);
  const calSRef=useRef(1.0);
  const calPhRef=useRef("idle");calPhRef.current=calPhase;
  const calBufRef=useRef([]);
  const lastGRef=useRef("RELAX");
  const gmapRef=useRef(DEFAULT_GMAP);
  const calRepRef=useRef(1);
  const calTRRef=useRef(5);
  const intRef=useRef(null);
  const sessionRef=useRef(null);
  const fatigueBaseRef=useRef(null);
  const hwModeRef=useRef(false);
  useEffect(()=>{hwModeRef.current=hwMode;},[hwMode]);

  useEffect(()=>{simGRef.current=simG;},[simG]);
  useEffect(()=>{uScaleRef.current=uScale;},[uScale]);
  useEffect(()=>{gmapRef.current=gestureMap;},[gestureMap]);
  useEffect(()=>{calRepRef.current=calRep;},[calRep]);
  useEffect(()=>{calTRRef.current=calTotalReps;},[calTotalReps]);

  // ── Init (model training runs async) ─────────────────
  useEffect(()=>{
    setLoading("Generating training data (15 users × 6 gestures)…");
    const t1=setTimeout(()=>{
      setLoading("Training Random Forest (30 trees, depth 5)…");
      const t2=setTimeout(()=>{
        setLoading("Building GNB + RF Ensemble…");
        const t3=setTimeout(()=>{
          const{model,meta,smoother}=getModel();
          modelRef.current=model;
          smootherRef.current=smoother;
          scalerRef.current=meta.scaler;
          setMeta(meta);setReady(true);
          (async()=>{
            try{
              const rVal=await storageGet("emg-cal-v5");if(rVal){const c=JSON.parse(rVal);setCalResult(c);calSRef.current=c.sf;}
              const mVal=await storageGet("emg-gmap");if(mVal){const g=JSON.parse(mVal);setGestureMap(g);gmapRef.current=g;}
              const pVal=await storageGet("emg-profiles");if(pVal){setProfiles(JSON.parse(pVal));}
            }catch(e){}
          })();
        },600);
        return()=>clearTimeout(t3);
      },400);
      return()=>clearTimeout(t2);
    },300);
    return()=>clearTimeout(t1);
  },[]);

  // Session timer
  useEffect(()=>{
    if(!live){clearInterval(sessionRef.current);return;}
    sessionRef.current=setInterval(()=>setSessionTime(t=>t+1),1000);
    return()=>clearInterval(sessionRef.current);
  },[live]);

  // ── Hardware WebSocket & WebSerial Functions ─────────────────
  const toggleHardwareWebSocket = useCallback(()=>{
    if(hwMode && wsRef.current){
      wsRef.current.close();
      setHwMode(false);
      setHwStatus("Offline (Simulated)");
      return;
    }
    setHwStatus("Connecting to ws://localhost:8765...");
    try{
      const ws=new WebSocket("ws://localhost:8765");
      ws.onopen=()=>{
        setHwMode(true);
        setHwStatus("Connected (ws://localhost:8765)");
        if(!live) setLive(true);
      };
      ws.onmessage=(e)=>{
        try{
          const data=JSON.parse(e.data);
          if(data.type==="emg_sample"){
            tick.current++;
            const pt={i:tick.current,v:+data.voltage.toFixed(4)};
            setSigData(p=>[...p,pt].slice(-DISP));
            bufRef.current.push(data.voltage);
            if(bufRef.current.length>WIN) bufRef.current.shift();
          }else if(data.type==="classification"||data.type==="gesture_event"){
            const g=data.gesture;
            const rms=data.rms||(g==="FIST"?0.75:g==="OPEN_HAND"?0.35:g==="DOUBLE_PULSE"?0.85:0.04);
            const conf=data.confidence?(data.confidence/100):0.95;
            const intensity=Math.min(100,Math.round(rms*150));
            setPred({g,conf,proba:{[g]:conf},intensity,snr:28,smoothed:true});

            // Generate real-time visual biopotential burst on the chart
            const burst=genWin(g, 1.0);
            const pts=burst.slice(0, 15).map((val, idx)=>({i:tick.current+idx, v:+val.toFixed(4)}));
            tick.current+=15;
            setSigData(p=>[...p,...pts].slice(-DISP));

            if(g!=="UNKNOWN"&&g!==lastGRef.current){
              lastGRef.current=g;
              doAction(g,intensity);
              setSessionLog(p=>[{id:tick.current,time:new Date().toLocaleTimeString(),gesture:g,conf:+(conf*100).toFixed(1),intensity,snr:28},...p].slice(0,200));
            }
          }
        }catch(err){}
      };
      ws.onerror=()=>{
        setHwStatus("Bridge Offline. Run: python hardware_dashboard_server.py");
        setHwMode(false);
      };
      ws.onclose=()=>{
        setHwMode(false);
        setHwStatus("Offline (Simulated)");
      };
      wsRef.current=ws;
    }catch(err){
      setHwStatus("WebSocket Error");
    }
  },[live,hwMode]);

  const connectWebSerial = useCallback(async ()=>{
    if(!navigator.serial){
      alert("WebSerial API is supported in Google Chrome, Microsoft Edge, and Opera!");
      return;
    }
    try{
      const port=await navigator.serial.requestPort();
      await port.open({baudRate:115200});
      setHwMode(true);
      setHwStatus("Connected (USB WebSerial)");
      if(!live) setLive(true);

      const textDecoder=new TextDecoderStream();
      port.readable.pipeTo(textDecoder.writable);
      const reader=textDecoder.readable.getReader();

      let lineBuf="";
      while(true){
        const{value,done}=await reader.read();
        if(done) break;
        lineBuf+=value;
        const lines=lineBuf.split("\n");
        lineBuf=lines.pop();

        for(const line of lines){
          const clean=line.trim();
          if(!clean) continue;
          const v=parseFloat(clean);
          if(!isNaN(v)){
            tick.current++;
            setSigData(p=>[...p,{i:tick.current,v:+v.toFixed(4)}].slice(-DISP));
            bufRef.current.push(v);
            if(bufRef.current.length>WIN) bufRef.current.shift();
          }
        }
      }
    }catch(err){
      console.log("WebSerial Error:",err);
    }
  },[live]);

  // ── Signal loop (Simulated generator active only when not in HW mode) ────────────────
  useEffect(()=>{
    if(!live||!ready)return;
    intRef.current=setInterval(()=>{
      if(!hwModeRef.current){
        const pts=[],sc=calSRef.current??uScaleRef.current;
        for(let s=0;s<5;s++){
          tick.current++;
          const v=emgSample(simGRef.current,sc,tick.current/SR);
          pts.push({i:tick.current,v:+v.toFixed(4)});
          bufRef.current.push(v);if(bufRef.current.length>WIN)bufRef.current.shift();
          if(calPhRef.current==="recording")calBufRef.current.push(v);
        }
        setSigData(p=>[...p,...pts].slice(-DISP));
      }

      if(tick.current%25===0&&bufRef.current.length>=WIN){
        const rawF=extractFeatures(bufRef.current);
        setRawFeat(rawF);

        // Apply scaler then calibration
        let fv=applyScaler(rawF,scalerRef.current||{mean:new Array(15).fill(0),std:new Array(15).fill(1)});
        if(calSRef.current&&calSRef.current!==1.0) fv=applyCalibration(fv,calSRef.current);

        // Ensemble prediction
        const rawProba=modelRef.current.proba(fv);

        // Temporal smoothing
        const{gesture:g,confidence:conf,proba}=smootherRef.current.update(rawProba);

        // Rejection class
        const finalG=conf>=CONF_THRESHOLD?g:"UNKNOWN";

        // Intensity from RMS
        const rms=rawF[2];
        const baseline=(calResult?.baseline||40)/1000;
        const maxAmp=calResult?Math.max(...Object.values(calResult.amps||CAL_EXP_RMS)):0.67;
        const intensity=Math.min(100,Math.max(0,Math.round((rms-baseline)/(maxAmp-baseline+0.01)*100)));

        // SNR
        const snr=Math.max(0,+(20*Math.log10(rms/(rawF[4]*0.2+0.001))).toFixed(1));

        // Fatigue
        if(finalG==="RELAX"){
          if(!fatigueBaseRef.current)fatigueBaseRef.current=rms;
          else if(rms>fatigueBaseRef.current*1.25)setFatigueWarn(true);
        }

        if(!hwModeRef.current){
          setPred({g:finalG,conf,proba,intensity,snr,smoothed:true});

          if(finalG!=="UNKNOWN"&&conf>0.65&&finalG!==lastGRef.current){
            lastGRef.current=finalG;
            doAction(finalG,intensity);
            setSessionLog(p=>[{id:tick.current,time:new Date().toLocaleTimeString(),
              gesture:finalG,conf:+(conf*100).toFixed(1),intensity,snr},...p].slice(0,200));
          }
        }
      }
    },10);
    return()=>clearInterval(intRef.current);
  },[live,ready,calResult]);

  // Cal phases
  useEffect(()=>{if(calPhase!=="get_ready")return;if(calReadyCnt<=0){setCalPhase("recording");return;}const t=setTimeout(()=>setCalReadyCnt(c=>c-1),1000);return()=>clearTimeout(t);},[calPhase,calReadyCnt]);
  useEffect(()=>{
    if(calPhase!=="recording")return;
    calBufRef.current=[];
    const t0=Date.now();
    const pi=setInterval(()=>setCalProg(Math.min(100,(Date.now()-t0)/REC_DUR*100)),60);
    const st=setTimeout(()=>{
      clearInterval(pi);setCalProg(100);
      const data=[...calBufRef.current];const f=extractFeatures(data);
      setCalRepData(prev=>[...prev,{rms:f[2],mav:f[0]}]);
      if(calRepRef.current<calTRRef.current)setCalPhase("rest");else setCalPhase("review");
    },REC_DUR);
    return()=>{clearInterval(pi);clearTimeout(st);};
  },[calPhase]);
  useEffect(()=>{if(calPhase!=="rest")return;const t=setTimeout(()=>{setCalRep(r=>r+1);calRepRef.current+=1;setCalReadyCnt(READY_SEC);setCalProg(0);setCalPhase("get_ready");},REST_DUR);return()=>clearTimeout(t);},[calPhase]);
  useEffect(()=>{
    if(calPhase!=="done")return;
    const amps={};
    calAllStats&&Object.entries(calAllStats).forEach(([g,st])=>{amps[g]=st.mean;});
    let rs=0,cnt=0;
    Object.entries(CAL_EXP_RMS).forEach(([g,ev])=>{if(amps[g]!=null&&amps[g]>0.001){rs+=amps[g]/ev;cnt++;}});
    const sf=cnt>0?+(rs/cnt).toFixed(3):1.0;
    const result={sf,baseline:+((amps.RELAX??0.04)*1000).toFixed(2),amps,allStats:calAllStats,quality:cnt>=5?"Excellent":cnt>=3?"Good":"Fair"};
    setCalResult(result);calSRef.current=result.sf;setCalPhase("complete");
    smootherRef.current?.reset();
    storageSet("emg-cal-v5",JSON.stringify(result));
  },[calPhase,calAllStats]);

  const doAction=useCallback((g,intensity=100)=>{
    const map=gmapRef.current;
    setDevs(prev=>{
      const next={...prev};
      Object.entries(map).forEach(([k,m])=>{
        if(m.on===g&&!next[k].on){next[k]={...next[k],on:true,intensity};addLog(`${next[k].ic} ${next[k].lbl} → ON (${intensity}%)`,"on");}
        if(m.off===g&&next[k].on){next[k]={...next[k],on:false,intensity:0};addLog(`${next[k].ic} ${next[k].lbl} → OFF`,"off");}
      });
      return next;
    });
  },[]);
  const addLog=useCallback((msg,type="info")=>setCmdLog(p=>[{t:new Date().toLocaleTimeString(),m:msg,type},...p].slice(0,25)),[]);

  const startCal=()=>{setCalAllStats({});setCalRepData([]);setCalStep(0);setCalRep(1);calRepRef.current=1;setCalReadyCnt(READY_SEC);setCalProg(0);const g=CAL_G[0];setSimG(g);simGRef.current=g;setCalPhase("intro");if(!live)setLive(true);};
  const beginReps=()=>{calRepRef.current=1;setCalRep(1);setCalReadyCnt(READY_SEC);setCalProg(0);setCalRepData([]);setCalPhase("get_ready");};
  const reviewCommit=(repData)=>{
    const g=CAL_G[calStep];
    const rmsVals=repData.map(r=>r.rms);
    const mean=rmsVals.reduce((a,b)=>a+b,0)/rmsVals.length;
    const std=rmsVals.length>1?Math.sqrt(rmsVals.reduce((s,r)=>s+(r-mean)**2,0)/rmsVals.length):0;
    const newStats={...calAllStats,[g]:{mean,std,cv:mean>0?(std/mean)*100:0,reps:rmsVals}};
    setCalAllStats(newStats);
    if(calStep<CAL_G.length-1){const ng=CAL_G[calStep+1];setCalStep(s=>s+1);setSimG(ng);simGRef.current=ng;setCalRepData([]);setCalRep(1);calRepRef.current=1;setCalReadyCnt(READY_SEC);setCalProg(0);setCalPhase("intro");}
    else{setCalAllStats(newStats);setCalPhase("done");}
  };
  const reviewRedo=()=>{setCalRepData([]);setCalRep(1);calRepRef.current=1;setCalReadyCnt(READY_SEC);setCalProg(0);setCalPhase("intro");};
  const saveProfile=()=>{
    if(!calResult||!profileName.trim())return;
    const upd={...profiles,[profileName]:calResult};setProfiles(upd);setActiveProfile(profileName);
    storageSet("emg-profiles",JSON.stringify(upd));
  };
  const loadProfile=(name)=>{const p=profiles[name];if(!p)return;setCalResult(p);calSRef.current=p.sf;setActiveProfile(name);smootherRef.current?.reset();};
  const exportCSV=()=>{const rows=[["Time","Gesture","Conf%","Intensity%","SNR"],...sessionLog.map(l=>[l.time,l.gesture,l.conf,l.intensity,l.snr])];const csv=rows.map(r=>r.join(",")).join("\n");const a=document.createElement("a");a.href="data:text/csv;charset=utf-8,"+encodeURIComponent(csv);a.download=`emg_${new Date().toISOString().slice(0,10)}.csv`;a.click();};
  const saveGMap=()=>{storageSet("emg-gmap",JSON.stringify(gestureMap));};
  const fmtTime=s=>`${String(Math.floor(s/60)).padStart(2,"0")}:${String(s%60).padStart(2,"0")}`;

  // ══════════════════════════════════════════════════════════
  // RENDER: MONITOR
  // ══════════════════════════════════════════════════════════
  const renderMonitor=()=>(
    <div style={{display:"flex",flexDirection:"column",gap:14}}>
      {fatigueWarn&&<div style={{background:"#fef3c7",border:"1px solid #fde68a",borderRadius:10,padding:"10px 14px",display:"flex",alignItems:"center",justifyContent:"space-between",gap:10}}>
        <span style={{fontSize:12,color:"#92400e"}}>⚠ <b>Muscle fatigue detected</b> — consider a short break or recalibrate.</span>
        <button onClick={()=>setFatigueWarn(false)} style={btn("ghost",{padding:"3px 10px",fontSize:11})}>✕</button>
      </div>}
      <div style={card()}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10,flexWrap:"wrap",gap:8}}>
          <div style={{display:"flex",alignItems:"center",gap:8,flexWrap:"wrap"}}>
            <div style={{width:8,height:8,borderRadius:"50%",background:hwMode?C.purple:live?C.green:C.t4,boxShadow:hwMode?`0 0 0 3px ${C.purpleL}`:live?`0 0 0 3px ${C.greenL}`:"none"}}/>
            <span style={{fontSize:12,fontWeight:700,color:C.t2}}>EMG Biopotential Stream · 15 Features</span>
            <span style={{...badge(hwMode?C.purple:C.t4,hwMode?C.purpleL:C.input),fontSize:10}}>
              {hwMode?"⚡ "+hwStatus:live?"Simulated Live":"Idle"}
            </span>
            {live&&<span style={{...badge(C.t4,C.input),fontSize:10}}>SNR {pred.snr} dB</span>}
            {live&&<span style={{fontSize:11,color:C.t3}}>⏱ {fmtTime(sessionTime)}</span>}
          </div>
          <div style={{display:"flex",gap:6,alignItems:"center"}}>
            <button onClick={toggleHardwareWebSocket}
              style={btn(hwMode?"danger":"primary",{padding:"6px 12px",fontSize:11,background:hwMode?C.red:C.purple})}>
              {hwMode?"⚡ Disconnect HW":"⚡ Connect Hardware (WS)"}
            </button>
            <button onClick={connectWebSerial}
              style={btn("secondary",{padding:"6px 12px",fontSize:11})}>
              🔌 USB WebSerial
            </button>
            <button onClick={()=>{setLive(l=>{if(l){setSigData([]);setSessionTime(0);smootherRef.current?.reset();fatigueBaseRef.current=null;}return!l;})}}
              style={btn(live?"danger":"success",{padding:"6px 18px"})}>
              {live?"■ Stop":"▶ Start"}
            </button>
          </div>
        </div>
        <div style={{background:C.input,border:`1px solid ${C.border}`,borderRadius:8,overflow:"hidden"}}>
          <ResponsiveContainer width="100%" height={148}>
            <AreaChart data={sigData} margin={{top:8,right:6,bottom:4,left:-18}}>
              <defs>
                <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={G_COLOR[pred.g]||C.blue} stopOpacity={0.2}/>
                  <stop offset="95%" stopColor={G_COLOR[pred.g]||C.blue} stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke={C.border}/>
              <YAxis domain={[-1.8,1.8]} tick={{fill:C.t4,fontSize:9}}/>
              <Tooltip contentStyle={{background:C.surf,border:`1px solid ${C.border}`,color:C.t2,fontSize:10,borderRadius:8}} formatter={v=>[v.toFixed(4)+"V","EMG"]} labelFormatter={()=>""}/>
              <Area type="monotone" dataKey="v" stroke={G_COLOR[pred.g]||C.blue} strokeWidth={2} fill="url(#sg)" dot={false} isAnimationActive={false}/>
            </AreaChart>
          </ResponsiveContainer>
        </div>
        {live&&<IntensityBar value={pred.intensity} color={G_COLOR[pred.g]||C.blue} label="Contraction Intensity"/>}
        <div style={{display:"flex",justifyContent:"space-between",marginTop:5,fontSize:10,color:C.t4}}><span>← 400ms</span><span>now →</span></div>
      </div>

      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
        <div style={card()}>
          <SL>Simulate Gesture</SL>
          <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:6}}>
            {GESTURES.map(g=>(
              <button key={g} onClick={()=>setSimG(g)}
                style={{background:simG===g?G_COLOR[g]:G_BG[g],color:simG===g?"#fff":G_COLOR[g],
                  border:`2px solid ${simG===g?G_COLOR[g]:"transparent"}`,borderRadius:10,
                  padding:"8px 4px",cursor:"pointer",fontWeight:700,fontSize:11,
                  fontFamily:"inherit",textAlign:"center",lineHeight:1.4,transition:"all .15s",
                  boxShadow:simG===g?`0 2px 8px ${G_COLOR[g]}44`:"none"}}>
                <div style={{fontSize:20,marginBottom:2}}>{G_ICON[g]}</div>
                <div style={{fontSize:10}}>{G_LABEL[g]}</div>
              </button>
            ))}
          </div>
          <div style={{marginTop:12}}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
              <span style={{fontSize:11,color:C.t3}}>EMG Scale</span>
              <span style={{fontSize:11,fontWeight:700,color:C.blue}}>{uScale.toFixed(2)}×</span>
            </div>
            <input type="range" min={0.5} max={1.8} step={0.05} value={uScale} onChange={e=>setUScale(+e.target.value)} style={{width:"100%",accentColor:C.blue}}/>
          </div>
        </div>
        <div style={{...card(),display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",gap:8,background:G_BG[pred.g]}}>
          <SL>Predicted Gesture</SL>
          <div style={{fontSize:52,lineHeight:1}}>{G_ICON[pred.g]}</div>
          {pred.g==="UNKNOWN"&&<div style={{fontSize:10,color:C.red}}>Below {(CONF_THRESHOLD*100).toFixed(0)}% threshold</div>}
          <div style={{color:G_COLOR[pred.g],fontSize:16,fontWeight:800}}>{G_LABEL[pred.g]}</div>
          <div style={{width:"100%",height:8,background:"rgba(0,0,0,0.08)",borderRadius:4}}>
            <div style={{width:`${(pred.conf*100).toFixed(0)}%`,height:"100%",background:pred.g==="UNKNOWN"?C.t4:G_COLOR[pred.g],borderRadius:4,transition:"width .25s"}}/>
          </div>
          <div style={{fontSize:11,color:C.t3}}>
            Confidence: <b style={{color:pred.g==="UNKNOWN"?C.t4:G_COLOR[pred.g]}}>{(pred.conf*100).toFixed(1)}%</b>
            <span style={{color:C.t4,fontSize:10}}> (smoothed)</span>
          </div>
          {pred.g!=="UNKNOWN"&&<IntensityBar value={pred.intensity} color={G_COLOR[pred.g]} label="Strength"/>}
        </div>
      </div>

      <div style={card()}>
        <SL>Class Probabilities — Smoothed over {SMOOTH_WIN} frames</SL>
        {GESTURES.map(g=>{
          const p=(pred.proba[g]??0)*100;
          return(
            <div key={g} style={{display:"flex",alignItems:"center",gap:10,marginBottom:7}}>
              <div style={{width:96,fontSize:11,color:pred.g===g?G_COLOR[g]:C.t3,fontWeight:pred.g===g?700:500}}>{G_ICON[g]} {G_LABEL[g]}</div>
              <div style={{flex:1,background:C.input,borderRadius:4,height:9,overflow:"hidden"}}>
                <div style={{width:`${p.toFixed(1)}%`,height:"100%",background:G_COLOR[g],opacity:pred.g===g?1:0.4,transition:"width .25s",borderRadius:4}}/>
              </div>
              <div style={{width:40,fontSize:11,color:C.t4,textAlign:"right",fontVariantNumeric:"tabular-nums"}}>{p.toFixed(1)}%</div>
            </div>
          );
        })}
      </div>
    </div>
  );

  // ══════════════════════════════════════════════════════════
  // RENDER: ANALYTICS (upgraded with model comparison)
  // ══════════════════════════════════════════════════════════
  const renderAnalytics=()=>{
    if(!meta)return<div style={{color:C.t4,textAlign:"center",padding:40}}>Loading…</div>;
    const maxCM=Math.max(...GESTURES.flatMap(g=>GESTURES.map(g2=>meta.cm[g]?.[g2]??0)));
    const modelComp=[
      {name:"Gaussian NB",acc:meta.gnbAcc,color:"#64748b",desc:"Fast, independent features"},
      {name:"Random Forest",acc:meta.rfAcc,color:C.blue,desc:"30 trees, depth 5, √15 features"},
      {name:"GNB + RF Ensemble",acc:meta.ensAcc,color:C.green,desc:"Weighted by test accuracy"},
    ];
    return(
      <div style={{display:"flex",flexDirection:"column",gap:14}}>
        {/* Model accuracy comparison */}
        <div style={card()}>
          <SL>Model Comparison — Test Accuracy</SL>
          {modelComp.map(m=>(
            <div key={m.name} style={{marginBottom:10}}>
              <div style={{display:"flex",justifyContent:"space-between",marginBottom:4}}>
                <div>
                  <span style={{fontSize:12,fontWeight:700,color:C.t2}}>{m.name}</span>
                  <span style={{fontSize:10,color:C.t4,marginLeft:8}}>{m.desc}</span>
                </div>
                <span style={{fontSize:14,fontWeight:800,color:m.color,fontVariantNumeric:"tabular-nums"}}>{(m.acc*100).toFixed(1)}%</span>
              </div>
              <div style={{background:C.input,borderRadius:4,height:12,overflow:"hidden"}}>
                <div style={{width:`${(m.acc*100).toFixed(1)}%`,height:"100%",background:m.color,borderRadius:4,transition:"width 1s"}}/>
              </div>
            </div>
          ))}
          <div style={{background:C.greenL,border:"1px solid #bbf7d0",borderRadius:8,padding:10,marginTop:8,fontSize:11,color:C.green}}>
            ✓ Ensemble is <b>{((meta.ensAcc-meta.gnbAcc)*100).toFixed(1)}%</b> more accurate than GNB alone and <b>{((meta.ensAcc-meta.rfAcc)*100).toFixed(1)}%</b> vs RF alone
          </div>
        </div>

        {/* KPI row */}
        <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10}}>
          {[["Ensemble Test Acc",`${(meta.teAcc*100).toFixed(1)}%`,C.green,C.greenL],
            ["Features",`${meta.nFeatures}`,C.blue,C.blueL],
            ["RF Trees",`${meta.nTrees}`,C.purple,C.purpleL],
            ["Smooth Window",`${meta.smoothWin} frames`,C.amber,C.amberL]].map(([l,v,c,bg])=>(
            <div key={l} style={{...card({background:bg,border:`1px solid ${c}22`}),textAlign:"center"}}>
              <div style={{fontSize:9,color:C.t4,marginBottom:4,fontWeight:700}}>{l}</div>
              <div style={{fontSize:20,fontWeight:900,color:c}}>{v}</div>
            </div>
          ))}
        </div>

        {/* Feature importance */}
        <div style={card()}>
          <SL>Feature Importance (GNB inter-class variance)</SL>
          <div style={{display:"flex",flexDirection:"column",gap:5}}>
            {(meta.featureImportance||[]).slice(0,12).map((f,i)=>(
              <div key={f.name} style={{display:"flex",alignItems:"center",gap:8}}>
                <div style={{width:88,fontSize:10,color:i<3?C.blue:C.t3,fontWeight:i<3?700:400}}>{f.name}</div>
                <div style={{flex:1,background:C.input,borderRadius:3,height:8,overflow:"hidden"}}>
                  <div style={{width:`${(f.importance/(meta.featureImportance[0]?.importance||1)*100).toFixed(0)}%`,height:"100%",
                    background:i<3?C.blue:i<6?C.purple:C.t4,borderRadius:3}}/>
                </div>
                <div style={{width:42,fontSize:10,color:C.t4,textAlign:"right",fontVariantNumeric:"tabular-nums"}}>{(f.importance*100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>

        {/* Confusion matrix */}
        <div style={card()}>
          <SL>Confusion Matrix — Ensemble (Test Set)</SL>
          <div style={{overflowX:"auto"}}>
            <table style={{borderCollapse:"collapse",width:"100%",fontSize:10}}>
              <thead><tr>
                <th style={{padding:"4px 6px",color:C.t4,textAlign:"right",width:72,fontWeight:500,fontSize:9}}>True↓ Pred→</th>
                {GESTURES.map(g=><th key={g} style={{padding:"4px 6px",color:C.t3,textAlign:"center",minWidth:52,fontWeight:600,fontSize:9}}>{G_ICON[g]}<br/>{G_LABEL[g].split(" ")[0]}</th>)}
              </tr></thead>
              <tbody>
                {GESTURES.map(g=>(
                  <tr key={g}>
                    <td style={{padding:"4px 6px",color:G_COLOR[g],textAlign:"right",fontSize:9,fontWeight:700}}>{G_ICON[g]} {G_LABEL[g]}</td>
                    {GESTURES.map(g2=>{
                      const v=meta.cm[g]?.[g2]??0,ok=g===g2,r=maxCM>0?v/maxCM:0;
                      return<td key={g2} style={{padding:"4px 6px",textAlign:"center",fontWeight:ok?800:400,
                        background:ok?`rgba(22,163,74,${.08+r*.5})`:r>.06?`rgba(220,38,38,${r*.45})`:"transparent",
                        color:ok?C.green:v>0?C.red:C.t4,border:`1px solid ${C.border}`,fontSize:11,fontVariantNumeric:"tabular-nums"}}>{v}</td>;
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Per-class metrics */}
        <div style={card()}>
          <SL>Per-Class Precision & Recall</SL>
          {GESTURES.map(g=>{
            const tp=meta.cm[g]?.[g]??0;
            const tpFP=GESTURES.reduce((s,g2)=>s+(meta.cm[g2]?.[g]??0),0);
            const tpFN=GESTURES.reduce((s,g2)=>s+(meta.cm[g]?.[g2]??0),0);
            const prec=tpFP>0?tp/tpFP:0,rec=tpFN>0?tp/tpFN:0;
            const f1=prec+rec>0?2*prec*rec/(prec+rec):0;
            return(
              <div key={g} style={{display:"flex",alignItems:"center",gap:8,marginBottom:7}}>
                <div style={{width:88,fontSize:10,color:G_COLOR[g],fontWeight:600}}>{G_ICON[g]} {G_LABEL[g]}</div>
                <div style={{flex:1,display:"flex",flexDirection:"column",gap:2}}>
                  <div style={{background:C.input,borderRadius:3,height:6,overflow:"hidden"}}>
                    <div style={{width:`${(prec*100).toFixed(0)}%`,height:"100%",background:G_COLOR[g],borderRadius:3}}/>
                  </div>
                  <div style={{background:C.input,borderRadius:3,height:6,overflow:"hidden"}}>
                    <div style={{width:`${(rec*100).toFixed(0)}%`,height:"100%",background:G_COLOR[g],opacity:.5,borderRadius:3}}/>
                  </div>
                </div>
                <div style={{width:110,fontSize:9,color:C.t3,textAlign:"right",fontVariantNumeric:"tabular-nums"}}>
                  P:{(prec*100).toFixed(0)}% R:{(rec*100).toFixed(0)}% F1:{(f1*100).toFixed(0)}%
                </div>
              </div>
            );
          })}
          <div style={{fontSize:9,color:C.t4,marginTop:4}}>■ Precision ░ Recall</div>
        </div>
      </div>
    );
  };

  // Calibration + IoT + GestureMap + SessionLog — same as v4 (condensed)
  const renderCalibration=()=>{
    const stepG=CAL_G[calStep],gc=G_COLOR[stepG],gb=G_BG[stepG];
    const rmsVals=calRepData.map(r=>r.rms);
    const mean=rmsVals.length>0?rmsVals.reduce((a,b)=>a+b,0)/rmsVals.length:0;
    const std=rmsVals.length>1?Math.sqrt(rmsVals.reduce((s,r)=>s+(r-mean)**2,0)/rmsVals.length):0;
    const cv=mean>0?(std/mean)*100:0;
    const expRms=CAL_EXP_RMS[stepG];
    const ampOk=mean>expRms*0.2&&mean<expRms*3.5;
    const cColor=cv<15?C.green:cv<30?C.amber:C.red;
    const StepTracker=()=>(
      <div style={{display:"flex",alignItems:"center",gap:0,marginTop:14}}>
        {CAL_G.map((g,i)=>{const done=calAllStats[g]||i<calStep,active=i===calStep;return(<div key={g} style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",gap:4}}><div style={{display:"flex",alignItems:"center",width:"100%"}}>{i>0&&<div style={{flex:1,height:2,background:done?G_COLOR[g]:C.border}}/>}<div style={{width:28,height:28,borderRadius:"50%",display:"flex",alignItems:"center",justifyContent:"center",fontSize:13,fontWeight:700,flexShrink:0,transition:"all .3s",background:done?G_COLOR[g]:active?G_BG[g]:C.input,color:done?"#fff":active?G_COLOR[g]:C.t4,border:`2px solid ${done?G_COLOR[g]:active?G_COLOR[g]:C.border}`,boxShadow:active?`0 0 0 4px ${G_BG[g]}`:"none"}}>{done?"✓":G_ICON[g]}</div>{i<CAL_G.length-1&&<div style={{flex:1,height:2,background:done?G_COLOR[g]:C.border}}/>}</div><span style={{fontSize:8,color:active?G_COLOR[g]:done?C.t3:C.t4,fontWeight:active?700:400}}>{G_LABEL[g].split(" ")[0]}</span></div>);})}
      </div>
    );
    return(<div style={{display:"flex",flexDirection:"column",gap:14}}>
      <div style={card()}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start",gap:10,flexWrap:"wrap"}}>
          <div><div style={{fontSize:16,fontWeight:800,color:C.t,marginBottom:4}}>User Calibration</div><div style={{fontSize:12,color:C.t3}}>Perform each gesture <b style={{color:C.blue}}>{calTotalReps} times</b> — normalises all 15 features to your muscle signature.</div></div>
          {calResult&&<Chip label={`✓ ${activeProfile||"Saved"} — ${calResult.sf}× (${calResult.quality})`} color={C.green} bg={C.greenL}/>}
        </div>
        {calPhase!=="idle"&&calPhase!=="complete"&&<StepTracker/>}
      </div>
      {Object.keys(profiles).length>0&&calPhase==="idle"&&(<div style={card()}><SL>Saved Profiles</SL><div style={{display:"flex",gap:8,flexWrap:"wrap"}}>{Object.keys(profiles).map(name=>(<div key={name} style={{display:"flex",gap:4,alignItems:"center",background:activeProfile===name?C.blueL:C.input,borderRadius:8,padding:"6px 12px",border:`1px solid ${activeProfile===name?C.borderA:C.border}`}}><span style={{fontSize:12,fontWeight:activeProfile===name?700:500,color:activeProfile===name?C.blue:C.t2}}>👤 {name}</span><span style={{fontSize:10,color:C.t4}}>({profiles[name].sf}×)</span><button onClick={()=>loadProfile(name)} style={btn("primary",{padding:"2px 8px",fontSize:10})}>Load</button></div>))}</div></div>)}
      {calPhase==="idle"&&(<div style={{...card(),textAlign:"center",padding:32}}><div style={{fontSize:52,marginBottom:10}}>🎯</div><div style={{fontSize:17,fontWeight:800,color:C.t,marginBottom:16}}>Ready to calibrate</div><div style={{display:"flex",gap:8,justifyContent:"center",marginBottom:16}}>{[3,5,7,10].map(n=>(<button key={n} onClick={()=>setCalTotalReps(n)} style={btn(calTotalReps===n?"primary":"secondary",{padding:"8px 20px",fontSize:14,fontWeight:700})}>{n}×</button>))}</div><div style={{fontSize:11,color:C.t4,marginBottom:20}}>≈ <b style={{color:C.blue}}>{Math.ceil(calTotalReps*5.5*6/60)} min</b> total</div><button onClick={startCal} style={btn("primary",{padding:"11px 36px",fontSize:14,fontWeight:700})}>Begin Calibration →</button></div>)}
      {calPhase==="intro"&&(<div style={{...card({background:gb,border:`1px solid ${gc}22`}),textAlign:"center",padding:28}}><Chip label={`Gesture ${calStep+1} of ${CAL_G.length}`} color={gc} bg={C.surf}/><div style={{fontSize:62,margin:"16px 0 10px"}}>{G_ICON[stepG]}</div><div style={{fontSize:20,fontWeight:800,color:gc,marginBottom:8}}>{G_LABEL[stepG]}</div><div style={{fontSize:13,color:C.t3,marginBottom:20,maxWidth:340,margin:"0 auto 20px",lineHeight:1.7}}>{CAL_INST[stepG]}</div><button onClick={beginReps} style={btn("primary",{background:gc,padding:"11px 32px",fontSize:13,fontWeight:700})}>▶ Start {calTotalReps} Repetitions</button></div>)}
      {calPhase==="get_ready"&&(<div style={{...card({border:`2px solid ${gc}22`,background:gb}),textAlign:"center",padding:28}}><div style={{...badge(gc,C.surf),margin:"0 auto 16px"}}>Rep {calRep} of {calTotalReps}</div><div style={{fontSize:56,marginBottom:8}}>{G_ICON[stepG]}</div><div style={{fontSize:96,fontWeight:900,color:gc,lineHeight:1,fontVariantNumeric:"tabular-nums"}}>{calReadyCnt}</div><div style={{fontSize:12,color:C.t4,marginTop:12}}>Get into position…</div><div style={{marginTop:20}}><RepDots total={calTotalReps} current={calRep} phase={calPhase} color={gc}/></div></div>)}
      {calPhase==="recording"&&(<><div style={{...card({border:`2px solid ${gc}`,background:gb}),textAlign:"center",padding:24}}><div style={{display:"flex",alignItems:"center",justifyContent:"center",gap:8,marginBottom:12}}><div style={{width:10,height:10,borderRadius:"50%",background:C.red,animation:"pulse .7s infinite"}}/><span style={{fontSize:13,fontWeight:800,color:C.red}}>RECORDING — REP {calRep}/{calTotalReps}</span></div><div style={{fontSize:64,marginBottom:8}}>{G_ICON[stepG]}</div><div style={{fontSize:16,fontWeight:800,color:gc,marginBottom:6}}>{G_LABEL[stepG]}</div><div style={{background:"rgba(255,255,255,0.5)",borderRadius:10,height:14,overflow:"hidden",border:`1px solid ${gc}33`,marginBottom:6}}><div style={{width:`${calProg}%`,height:"100%",background:`linear-gradient(90deg,${gc},${C.purple})`,borderRadius:10,transition:"width .06s linear"}}/></div><div style={{fontSize:11,color:C.t4,marginBottom:16}}>{calProg.toFixed(0)}%</div><RepDots total={calTotalReps} current={calRep} phase={calPhase} color={gc}/></div><div style={card()}><SL>Live Signal</SL><div style={{background:C.input,borderRadius:8,border:`1px solid ${C.border}`,overflow:"hidden"}}><ResponsiveContainer width="100%" height={90}><AreaChart data={sigData} margin={{top:4,right:6,bottom:0,left:-20}}><defs><linearGradient id="rg" x1="0" y1="0" x2="0" y2="1"><stop offset="5%" stopColor={gc} stopOpacity={0.2}/><stop offset="95%" stopColor={gc} stopOpacity={0}/></linearGradient></defs><YAxis domain={[-1.8,1.8]} hide/><Area type="monotone" dataKey="v" stroke={gc} fill="url(#rg)" strokeWidth={2} dot={false} isAnimationActive={false}/></AreaChart></ResponsiveContainer></div></div></>)}
      {calPhase==="rest"&&(<div style={{...card(),textAlign:"center",padding:28}}><div style={{...badge(C.green,C.greenL),margin:"0 auto 14px"}}>Rep {calRep} complete ✓</div><div style={{fontSize:52,marginBottom:10}}>✋</div><div style={{fontSize:17,fontWeight:700,color:C.t,marginBottom:6}}>Relax your hand</div><div style={{fontSize:12,color:C.t3,marginBottom:16}}>Rest · Rep {calRep+1} starts in {REST_DUR/1000}s</div>{calRepData.length>0&&<div style={{display:"flex",gap:4,justifyContent:"center",marginBottom:16}}>{calRepData.map((r,i)=>{const h=Math.min(40,Math.max(6,Math.round((r.rms/expRms)*32)));return(<div key={i} style={{display:"flex",flexDirection:"column",alignItems:"center",gap:3}}><div style={{width:22,background:C.input,borderRadius:3,height:40,display:"flex",alignItems:"flex-end"}}><div style={{width:"100%",height:`${h}px`,background:gc,borderRadius:3}}/></div><span style={{fontSize:8,color:C.t4}}>R{i+1}</span></div>);})}</div>}<RepDots total={calTotalReps} current={calRep} phase={calPhase} color={gc}/></div>)}
      {calPhase==="review"&&(<><div style={{...card({background:ampOk?C.greenL:C.redL,border:`1px solid ${ampOk?"#bbf7d0":"#fecaca"}`}),textAlign:"center",padding:20}}><div style={{...badge(ampOk?C.green:C.red,C.surf),margin:"0 auto 10px"}}>{calTotalReps}/{calTotalReps} reps complete</div><div style={{fontSize:48,marginBottom:6}}>{G_ICON[stepG]}</div><div style={{fontSize:16,fontWeight:800,color:C.t}}>{G_LABEL[stepG]} — Summary</div></div><div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:10}}><div style={{...card(),textAlign:"center"}}><div style={{fontSize:10,color:C.t4,marginBottom:3,fontWeight:600}}>Mean RMS</div><div style={{fontSize:22,fontWeight:900,color:C.blue,fontVariantNumeric:"tabular-nums"}}>{(mean*1000).toFixed(1)}<span style={{fontSize:12}}> mV</span></div><div style={{fontSize:10,color:C.t4}}>Expected ~{(expRms*1000).toFixed(0)} mV</div></div><div style={{...card(),textAlign:"center"}}><div style={{fontSize:10,color:C.t4,marginBottom:3,fontWeight:600}}>Consistency CV</div><div style={{fontSize:22,fontWeight:900,color:cColor,fontVariantNumeric:"tabular-nums"}}>{cv.toFixed(1)}<span style={{fontSize:12}}>%</span></div><div style={{fontSize:10,color:cColor,fontWeight:700}}>{cv<15?"Excellent":cv<30?"Good":cv<50?"Fair":"Inconsistent"}</div></div><div style={{...card({background:ampOk?C.greenL:C.redL}),textAlign:"center"}}><div style={{fontSize:10,color:C.t4,marginBottom:3,fontWeight:600}}>Amplitude</div><div style={{fontSize:15,fontWeight:800,color:ampOk?C.green:C.red,marginTop:4}}>{ampOk?"✓ Good":mean<expRms*0.2?"Too Weak":"Too Strong"}</div></div></div><div style={card()}><SL>Amplitude per rep</SL><ResponsiveContainer width="100%" height={120}><BarChart data={calRepData.map((r,i)=>({rep:`R${i+1}`,rms:+(r.rms*1000).toFixed(2)}))} margin={{top:6,right:10,bottom:0,left:0}}><CartesianGrid strokeDasharray="3 3" stroke={C.border}/><XAxis dataKey="rep" tick={{fill:C.t4,fontSize:10}}/><YAxis tick={{fill:C.t4,fontSize:9}} unit="mV" domain={[0,"auto"]}/><Tooltip contentStyle={{background:C.surf,border:`1px solid ${C.border}`,color:C.t2,fontSize:10,borderRadius:8}} formatter={v=>[`${v} mV`,"RMS"]}/><Bar dataKey="rms" fill={gc} radius={[4,4,0,0]} isAnimationActive={false}/></BarChart></ResponsiveContainer></div><div style={{display:"flex",gap:8}}><button onClick={reviewRedo} style={btn("secondary")}>↩ Redo</button><button onClick={()=>reviewCommit(calRepData)} style={btn("primary",{flex:1,background:calStep<CAL_G.length-1?gc:C.green})}>{calStep<CAL_G.length-1?`Next: ${G_LABEL[CAL_G[calStep+1]]} →`:"✓ Finish Calibration"}</button></div></>)}
      {calPhase==="complete"&&calResult&&(<><div style={{...card(),textAlign:"center",padding:28}}><div style={{fontSize:48,marginBottom:8}}>🎉</div><div style={{fontSize:18,fontWeight:800,color:C.green,marginBottom:4}}>Calibration Complete!</div><div style={{fontSize:12,color:C.t3}}>Scaling factor: <b style={{color:C.blue}}>{calResult.sf}×</b> · Quality: <b style={{color:calResult.quality==="Excellent"?C.green:C.amber}}>{calResult.quality}</b></div></div><div style={card()}><SL>Save as Named Profile</SL><div style={{display:"flex",gap:8,marginBottom:10}}><input value={profileName} onChange={e=>setProfileName(e.target.value)} placeholder="Enter profile name" style={{flex:1,padding:"8px 12px",border:`1px solid ${C.border}`,borderRadius:8,fontSize:13,fontFamily:"inherit"}}/><button onClick={saveProfile} style={btn("primary")}>💾 Save Profile</button></div><div style={{fontSize:11,color:C.t3}}>Saved profiles load instantly — no recalibration needed for the same user.</div></div><div style={{display:"flex",gap:8}}><button onClick={()=>{setCalPhase("idle");setCalAllStats({});setCalRepData([]);calSRef.current=1.0;setCalResult(null);setActiveProfile(null);}} style={btn("secondary")}>🔄 Recalibrate</button><button onClick={()=>setTab("monitor")} style={btn("primary",{flex:1})}>▶ Start Recognition</button></div></>)}
    </div>);
  };

  const renderGestureMap=()=>(<div style={{display:"flex",flexDirection:"column",gap:14}}><div style={card()}><div style={{display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:10}}><div><div style={{fontSize:15,fontWeight:800,color:C.t,marginBottom:4}}>Custom Gesture Mapping</div><div style={{fontSize:12,color:C.t3}}>Assign any gesture to any device ON/OFF trigger.</div></div><div style={{display:"flex",gap:8}}><button onClick={()=>{setGestureMap(DEFAULT_GMAP);gmapRef.current=DEFAULT_GMAP;}} style={btn("secondary",{fontSize:11})}>Reset</button><button onClick={saveGMap} style={btn("primary",{fontSize:11})}>💾 Save</button></div></div></div>{Object.entries(devs).map(([k,d])=>(<div key={k} style={card()}><div style={{display:"flex",alignItems:"center",gap:10,marginBottom:14}}><div style={{fontSize:28}}>{d.ic}</div><div style={{flex:1}}><div style={{fontSize:14,fontWeight:700,color:C.t}}>{d.lbl}</div><div style={{fontSize:11,color:C.t4}}>ON: {gestureMap[k]?.on?`${G_ICON[gestureMap[k].on]} ${G_LABEL[gestureMap[k].on]}`:"—"} · OFF: {gestureMap[k]?.off?`${G_ICON[gestureMap[k].off]} ${G_LABEL[gestureMap[k].off]}`:"—"}</div></div><div style={{...badge(d.on?C.green:C.t4,d.on?C.greenL:C.input)}}>{d.on?"● ON":"○ OFF"}</div></div><div style={{marginBottom:10}}><div style={{fontSize:11,fontWeight:600,color:C.t3,marginBottom:6}}>🔼 Turn <b style={{color:C.green}}>ON</b> when I do:</div><div style={{display:"flex",flexWrap:"wrap",gap:6}}><NoPill selected={!gestureMap[k]?.on} onClick={()=>setGestureMap(m=>({...m,[k]:{...m[k],on:null}}))}/>{GESTURES.map(g=><GesturePill key={g} g={g} selected={gestureMap[k]?.on===g} onClick={()=>setGestureMap(m=>({...m,[k]:{...m[k],on:g}}))}/>)}</div></div><div><div style={{fontSize:11,fontWeight:600,color:C.t3,marginBottom:6}}>🔽 Turn <b style={{color:C.red}}>OFF</b> when I do:</div><div style={{display:"flex",flexWrap:"wrap",gap:6}}><NoPill selected={!gestureMap[k]?.off} onClick={()=>setGestureMap(m=>({...m,[k]:{...m[k],off:null}}))}/>{GESTURES.map(g=><GesturePill key={g} g={g} selected={gestureMap[k]?.off===g} onClick={()=>setGestureMap(m=>({...m,[k]:{...m[k],off:g}}))}/>)}</div></div></div>))}</div>);

  const renderIoT=()=>(<div style={{display:"flex",flexDirection:"column",gap:14}}><div style={{...card(),display:"flex",alignItems:"center",gap:12}}><div style={{width:9,height:9,borderRadius:"50%",background:live?C.green:C.t4,boxShadow:live?`0 0 0 3px ${C.greenL}`:"none"}}/><div><div style={{fontSize:12,fontWeight:600,color:C.t2}}>MQTT · mqtt.emg-iot.local:1883</div><div style={{fontSize:10,color:C.t4}}>Topic: emg/gesture/recognized · QoS 1</div></div><div style={{marginLeft:"auto",...badge(live?C.green:C.t4,live?C.greenL:C.input)}}>ESP32 {live?"ONLINE":"OFFLINE"}</div></div><div style={{...card({background:G_BG[pred.g],border:`1px solid ${G_COLOR[pred.g]}22`}),display:"flex",gap:14,alignItems:"center"}}><div style={{fontSize:38}}>{G_ICON[pred.g]}</div><div><div style={{fontSize:9,color:C.t4,fontWeight:700,letterSpacing:"0.1em",marginBottom:3}}>MQTT PUBLISH</div><div style={{fontSize:11,fontFamily:"monospace",color:C.t3}}>{`{"g":"${pred.g}","c":${(pred.conf*100).toFixed(0)},"intensity":${pred.intensity}}`}</div><div style={{fontSize:13,fontWeight:700,color:G_COLOR[pred.g],marginTop:2}}>{G_LABEL[pred.g]} · {(pred.conf*100).toFixed(1)}% · {pred.intensity}% strength</div></div></div><div style={{display:"grid",gridTemplateColumns:"repeat(3,1fr)",gap:10}}>{Object.entries(devs).map(([k,d])=>(<div key={k} onClick={()=>{setDevs(p=>({...p,[k]:{...p[k],on:!p[k].on,intensity:p[k].on?0:100}}));addLog(`${d.ic} ${d.lbl} → ${d.on?"OFF":"ON"} (manual)`,d.on?"off":"on");}} style={{...card({padding:14}),cursor:"pointer",transition:"all .2s",textAlign:"center",background:d.on?C.greenL:C.surf,border:`1px solid ${d.on?"#86efac":C.border}`,boxShadow:d.on?"0 2px 12px rgba(22,163,74,0.15)":"0 1px 4px rgba(0,0,0,.06)"}}><div style={{fontSize:26,marginBottom:5,filter:d.on?"none":"grayscale(.5) opacity(.6)",transition:"filter .3s"}}>{d.ic}</div><div style={{fontSize:11,fontWeight:700,color:C.t2,marginBottom:4}}>{d.lbl}</div><div style={{...badge(d.on?C.green:C.t4,d.on?"rgba(22,163,74,0.15)":"rgba(0,0,0,0.05)"),margin:"0 auto 8px"}}>{d.on?"● ON":"○ OFF"}</div>{d.on&&<IntensityBar value={d.intensity||100} color={C.green}/>}</div>))}</div><div style={card()}><SL>Command Log</SL>{cmdLog.length===0?<div style={{color:C.t4,fontSize:11,textAlign:"center",padding:12}}>No commands yet.</div>:<div style={{maxHeight:200,overflowY:"auto",display:"flex",flexDirection:"column",gap:4}}>{cmdLog.map((l,i)=>(<div key={i} style={{display:"flex",gap:10,padding:"5px 0",borderBottom:`1px solid ${C.border}`}}><span style={{color:C.t4,fontSize:10,minWidth:66}}>{l.t}</span><span style={{fontSize:11,color:l.type==="on"?C.green:l.type==="off"?C.red:C.t2,fontWeight:l.type==="on"||l.type==="off"?600:400}}>{l.m}</span></div>))}</div>}</div></div>);

  const renderLog=()=>{
    const counts={};GESTURES.forEach(g=>{counts[g]=sessionLog.filter(l=>l.gesture===g).length;});
    counts["UNKNOWN"]=sessionLog.filter(l=>l.gesture==="UNKNOWN").length;
    const pieData=Object.entries(counts).filter(([,v])=>v>0).map(([g,v])=>({name:G_LABEL[g],value:v,color:G_COLOR[g]}));
    return(<div style={{display:"flex",flexDirection:"column",gap:14}}>
      <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:10}}>
        {[["Duration",fmtTime(sessionTime),C.blue],["Predictions",sessionLog.length,C.green],["Avg Conf",sessionLog.length>0?`${(sessionLog.reduce((s,l)=>s+l.conf,0)/sessionLog.length).toFixed(1)}%`:"—",C.purple],["Rejected",sessionLog.filter(l=>l.gesture==="UNKNOWN").length,C.red]].map(([l,v,c])=>(<div key={l} style={{...card(),textAlign:"center"}}><div style={{fontSize:9,color:C.t4,marginBottom:4,fontWeight:700,textTransform:"uppercase"}}>{l}</div><div style={{fontSize:22,fontWeight:900,color:c}}>{v}</div></div>))}
      </div>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:14}}>
        <div style={card()}><SL>Gesture Distribution</SL>{pieData.length===0?<div style={{color:C.t4,fontSize:11,textAlign:"center",padding:24}}>No data yet.</div>:(<><ResponsiveContainer width="100%" height={160}><PieChart><Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={65} isAnimationActive={false}>{pieData.map((d,i)=><Cell key={i} fill={d.color}/>)}</Pie><Tooltip contentStyle={{fontSize:11,borderRadius:8}} formatter={(v,n)=>[`${v} predictions`,n]}/></PieChart></ResponsiveContainer><div style={{display:"flex",flexWrap:"wrap",gap:6}}>{pieData.map(d=>(<div key={d.name} style={{display:"flex",alignItems:"center",gap:4,fontSize:10}}><div style={{width:8,height:8,borderRadius:"50%",background:d.color}}/><span style={{color:C.t3}}>{d.name}: <b style={{color:d.color}}>{d.value}</b></span></div>))}</div></>)}</div>
        <div style={card()}><SL>Intensity Over Session</SL>{sessionLog.length<2?<div style={{color:C.t4,fontSize:11,textAlign:"center",padding:24}}>Need 2+ predictions.</div>:(<ResponsiveContainer width="100%" height={180}><LineChart data={[...sessionLog].reverse().map((l,i)=>({i:i+1,intensity:l.intensity}))} margin={{top:5,right:5,bottom:0,left:-20}}><CartesianGrid strokeDasharray="3 3" stroke={C.border}/><XAxis dataKey="i" tick={{fill:C.t4,fontSize:9}}/><YAxis domain={[0,100]} tick={{fill:C.t4,fontSize:9}} unit="%"/><Tooltip contentStyle={{fontSize:10,borderRadius:8}} formatter={v=>[`${v}%`,"Intensity"]}/><Line type="monotone" dataKey="intensity" stroke={C.blue} strokeWidth={2} dot={false} isAnimationActive={false}/></LineChart></ResponsiveContainer>)}</div>
      </div>
      <div style={card()}>
        <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",marginBottom:12}}>
          <SL>Prediction History ({sessionLog.length})</SL>
          <div style={{display:"flex",gap:8}}><button onClick={()=>setSessionLog([])} style={btn("secondary",{fontSize:11,padding:"5px 12px"})}>Clear</button><button onClick={exportCSV} style={btn("primary",{fontSize:11,padding:"5px 12px"})}>⬇ CSV</button></div>
        </div>
        {sessionLog.length===0?<div style={{color:C.t4,fontSize:11,textAlign:"center",padding:20}}>No predictions yet.</div>:(<div style={{maxHeight:320,overflowY:"auto"}}><table style={{width:"100%",borderCollapse:"collapse",fontSize:11}}><thead style={{position:"sticky",top:0}}><tr>{["Time","Gesture","Confidence","Intensity","SNR"].map(h=>(<th key={h} style={{padding:"7px 10px",fontSize:10,fontWeight:700,color:C.t4,letterSpacing:"0.07em",textTransform:"uppercase",textAlign:"left",background:C.bg,borderBottom:`1px solid ${C.border}`}}>{h}</th>))}</tr></thead><tbody>{sessionLog.map((l,i)=>(<tr key={l.id} style={{background:i%2===0?C.surf:"#f8fafc"}}><td style={{padding:"6px 10px",color:C.t4,fontVariantNumeric:"tabular-nums"}}>{l.time}</td><td style={{padding:"6px 10px"}}><span style={{fontWeight:700,color:G_COLOR[l.gesture]||C.t4}}>{G_ICON[l.gesture]} {G_LABEL[l.gesture]}</span></td><td style={{padding:"6px 10px"}}><div style={{display:"flex",alignItems:"center",gap:6}}><div style={{width:50,height:6,background:C.border,borderRadius:3}}><div style={{width:`${l.conf}%`,height:"100%",background:G_COLOR[l.gesture]||C.t4,borderRadius:3}}/></div><span style={{color:C.t3}}>{l.conf}%</span></div></td><td style={{padding:"6px 10px"}}><div style={{display:"flex",alignItems:"center",gap:6}}><div style={{width:40,height:6,background:C.border,borderRadius:3}}><div style={{width:`${l.intensity}%`,height:"100%",background:C.blue,borderRadius:3}}/></div><span style={{color:C.t3}}>{l.intensity}%</span></div></td><td style={{padding:"6px 10px",color:C.t3,fontVariantNumeric:"tabular-nums"}}>{l.snr} dB</td></tr>))}</tbody></table></div>)}
      </div>
    </div>);
  };

  // ══════════════════════════════════════════════════════════
  // LOADING SCREEN
  // ══════════════════════════════════════════════════════════
  if(!ready)return(
    <div style={{background:C.bg,minHeight:"100vh",display:"flex",alignItems:"center",justifyContent:"center",fontFamily:"'Segoe UI',system-ui,sans-serif"}}>
      <div style={{textAlign:"center",maxWidth:340}}>
        <div style={{fontSize:44,marginBottom:14}}>🧠</div>
        <div style={{fontSize:14,fontWeight:800,color:C.t,marginBottom:4}}>{loading}</div>
        <div style={{fontSize:11,color:C.t3,marginBottom:20}}>15 features · GNB + Random Forest (30 trees) · Weighted Ensemble · Z-score normalisation</div>
        <div style={{width:"100%",height:6,background:C.border,borderRadius:3,overflow:"hidden"}}>
          <div style={{height:"100%",background:`linear-gradient(90deg,${C.blue},${C.purple})`,borderRadius:3,animation:"prog 1.4s ease-in-out infinite alternate"}}/>
        </div>
        <div style={{marginTop:16,display:"flex",flexDirection:"column",gap:4}}>
          {[["GNB alone","~85% accuracy"],["Random Forest 30T","~90% accuracy"],["GNB + RF Ensemble","~92-95% accuracy"]].map(([m,a])=>(
            <div key={m} style={{display:"flex",justifyContent:"space-between",fontSize:11,padding:"4px 8px",background:C.surf,borderRadius:6,border:`1px solid ${C.border}`}}>
              <span style={{color:C.t3}}>{m}</span><span style={{color:C.green,fontWeight:700}}>{a}</span>
            </div>
          ))}
        </div>
      </div>
      <style>{`@keyframes prog{from{width:10%}to{width:95%}}`}</style>
    </div>
  );

  // ══════════════════════════════════════════════════════════
  // MAIN
  // ══════════════════════════════════════════════════════════
  const TABS=[["monitor","📡","Live Monitor"],["calibration","🎯","Calibration"],["map","🎮","Gesture Map"],["iot","🏠","IoT Control"],["log","📋","Session Log"],["analytics","📊","Analytics"]];
  return(
    <div style={{background:C.bg,minHeight:"100vh",fontFamily:"'Inter','Segoe UI',-apple-system,system-ui,sans-serif",color:C.t2}}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');*{font-family:'Inter','Segoe UI',system-ui,sans-serif;}@keyframes pulse{0%,100%{opacity:1}50%{opacity:.25}}@keyframes pop{0%{transform:scale(.6);opacity:.4}70%{transform:scale(1.1)}100%{transform:scale(1);opacity:1}}`}</style>
      <div style={{background:C.surf,borderBottom:`1px solid ${C.border}`,padding:"10px 20px",display:"flex",alignItems:"center",gap:12,position:"sticky",top:0,zIndex:100,boxShadow:"0 1px 4px rgba(0,0,0,.06)"}}>
        <div style={{width:36,height:36,background:C.blueL,borderRadius:10,display:"flex",alignItems:"center",justifyContent:"center",fontSize:20,border:`1px solid ${C.borderA}`}}>💪</div>
        <div><div style={{fontWeight:800,fontSize:15,color:C.t}}>EMG Gesture Control <span style={{fontSize:11,fontWeight:500,color:C.t4}}>v5 — Enhanced ML</span></div><div style={{fontSize:10,color:C.t4}}>15 Features · GNB + RF Ensemble · Temporal Smoothing · Z-score Normalisation</div></div>
        <div style={{marginLeft:"auto",display:"flex",gap:8,alignItems:"center",flexWrap:"wrap"}}>
          {meta&&<Chip label={`Ensemble ${(meta.teAcc*100).toFixed(1)}%`} color={C.green} bg={C.greenL}/>}
          {meta&&<Chip label={`RF ${(meta.rfAcc*100).toFixed(1)}%`} color={C.blue} bg={C.blueL}/>}
          {activeProfile&&<Chip label={`👤 ${activeProfile}`} color={C.purple} bg={C.purpleL}/>}
          {calResult&&!activeProfile&&<Chip label={`Cal ${calResult.sf}×`} color={C.amber} bg={C.amberL}/>}
          <Chip label={live?"● Live":"○ Idle"} color={live?C.green:C.t4} bg={live?C.greenL:C.input}/>
        </div>
      </div>
      <div style={{background:C.surf,borderBottom:`1px solid ${C.border}`,display:"flex",overflowX:"auto"}}>
        {TABS.map(([id,ic,lb])=>(<button key={id} onClick={()=>setTab(id)} style={{background:"none",border:"none",borderBottom:`2.5px solid ${tab===id?C.blue:"transparent"}`,color:tab===id?C.blue:C.t3,padding:"10px 16px",cursor:"pointer",fontSize:12,fontWeight:tab===id?700:500,display:"flex",alignItems:"center",gap:5,whiteSpace:"nowrap",transition:"all .15s",fontFamily:"inherit"}}>{ic} {lb}{id==="log"&&sessionLog.length>0&&<span style={{background:C.amber,color:"#fff",borderRadius:10,padding:"1px 6px",fontSize:9,fontWeight:700}}>{sessionLog.length}</span>}</button>))}
      </div>
      <div style={{padding:"16px 16px 32px",maxWidth:880,margin:"0 auto"}}>
        {tab==="monitor"    &&renderMonitor()}
        {tab==="calibration"&&renderCalibration()}
        {tab==="map"        &&renderGestureMap()}
        {tab==="iot"        &&renderIoT()}
        {tab==="log"        &&renderLog()}
        {tab==="analytics"  &&renderAnalytics()}
      </div>
    </div>
  );
}
