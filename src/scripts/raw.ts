import { buildAtlas, DENSITY } from "./ascii";

// Renderer shader "brut" — pleine résolution, sans pipeline ASCII.
// Scenes : 0 liquide (metaballs 3D raymarchees), 1 menger (fractale 3D),
// 2 flux (domain warping), 3 caustiques. La souris pilote lumiere / warp.

const VS = `attribute vec2 aPos; varying vec2 vUV;
void main(){ vUV = aPos*0.5+0.5; gl_Position = vec4(aPos,0.,1.); }`;

const FS = `
precision highp float;
varying vec2 vUV;
uniform vec2 uRes; uniform float uTime; uniform int uScene; uniform vec2 uMouse;

const int MAX_TRAIL = 16;
uniform int uTrailCount;
uniform vec2 uTrailPos[MAX_TRAIL];
uniform float uTrailStart[MAX_TRAIL];
uniform float uNow;
uniform vec2 uTrailDir; // direction de deplacement lissee (norme ~ vitesse)
uniform sampler2D uAtlas; uniform vec2 uAtlasGrid; uniform float uGlyphCount; uniform int uHasAtlas;
uniform float uSeed;      // graine par instance (marbrure unique par article)
uniform float uTrailRad;  // rayon de la trainee (fraction du min de l'ecran)
uniform sampler2D uTitle; uniform vec4 uTitleRect; uniform float uMelt; uniform int uHasTitle;

vec3 magmaPal(float x){
  vec3 c1=vec3(0.03,0.01,0.07), c2=vec3(0.38,0.06,0.55), c3=vec3(1.0,0.34,0.10), c4=vec3(1.0,0.82,0.30), c5=vec3(1.0,0.97,0.85);
  x=clamp(x,0.,1.);
  if(x<0.25) return mix(c1,c2,x/0.25);
  if(x<0.55) return mix(c2,c3,(x-0.25)/0.30);
  if(x<0.85) return mix(c3,c4,(x-0.55)/0.30);
  return mix(c4,c5,(x-0.85)/0.15);
}
float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x), f.y); }
float fbm(vec2 p){ float v=0.,a=.5; for(int k=0;k<5;k++){ v+=a*noise(p); p=p*2.03+11.7; a*=.5; } return v; }

// ---------- 0 · LIQUIDE : metaballs raymarchees, metal en fusion ----------
float smin(float a, float b, float k){ float h=clamp(0.5+0.5*(b-a)/k,0.,1.); return mix(b,a,h)-k*h*(1.0-h); }
float mapLiquid(vec3 p, float t){
  float d=1e9;
  for(int i=0;i<6;i++){
    float fi=float(i);
    vec3 c=vec3(
      1.1*sin(t*(0.4+fi*0.06)+fi*2.1),
      0.8*cos(t*(0.33+fi*0.05)+fi*1.3),
      0.6*sin(t*(0.27+fi*0.07)+fi*3.7));
    d=smin(d, length(p-c)-(0.42+0.1*sin(t*0.8+fi)), 0.55);
  }
  return d;
}
vec3 normLiquid(vec3 p, float t){
  vec2 e=vec2(0.001,0.0);
  return normalize(vec3(
    mapLiquid(p+e.xyy,t)-mapLiquid(p-e.xyy,t),
    mapLiquid(p+e.yxy,t)-mapLiquid(p-e.yxy,t),
    mapLiquid(p+e.yyx,t)-mapLiquid(p-e.yyx,t)));
}
vec3 sceneLiquid(vec2 uv, float t){
  vec2 p=(uv-0.5)*vec2(uRes.x/uRes.y,1.0)*2.0;
  vec3 ro=vec3(0.,0.,3.6), rd=normalize(vec3(p,-2.0));
  float dist=0.0; float d;
  for(int i=0;i<72;i++){
    d=mapLiquid(ro+rd*dist,t);
    if(d<0.001||dist>9.0) break;
    dist+=d;
  }
  vec3 bg=mix(vec3(0.02,0.01,0.05), vec3(0.10,0.02,0.12), pow(1.0-abs(uv.y-0.5)*2.0,2.0));
  bg+=vec3(0.25,0.05,0.20)*exp(-length(p)*1.2)*0.6;
  if(dist>9.0) return bg;
  vec3 pos=ro+rd*dist, n=normLiquid(pos,t);
  vec3 lightDir=normalize(vec3((uMouse.x-0.5)*2.5,(0.5-uMouse.y)*2.5,1.2));
  float dif=max(0.,dot(n,lightDir));
  float fre=pow(1.0-max(0.,dot(n,-rd)),3.0);
  vec3 ref=reflect(rd,n);
  float spec=pow(max(0.,dot(ref,lightDir)),24.0);
  vec3 env=magmaPal(0.35+0.5*ref.y+0.15*sin(t*0.2));
  vec3 col=magmaPal(0.18+0.35*dif)*0.55 + env*fre*0.9 + vec3(1.0,0.9,0.7)*spec*1.2;
  return col;
}

// ---------- 1 · MENGER : eponge fractale raymarchee ----------
float sdBox(vec3 p, vec3 b){ vec3 q=abs(p)-b; return length(max(q,0.))+min(max(q.x,max(q.y,q.z)),0.); }
float mapMenger(vec3 p, float t){
  p.xz=mat2(cos(t*0.1),-sin(t*0.1),sin(t*0.1),cos(t*0.1))*p.xz;
  p.xy=mat2(cos(t*0.07),-sin(t*0.07),sin(t*0.07),cos(t*0.07))*p.xy;
  float d=sdBox(p,vec3(1.0));
  float s=1.0;
  for(int m=0;m<4;m++){
    vec3 a=mod(p*s,2.0)-1.0;
    s*=3.0;
    vec3 r=abs(1.0-3.0*abs(a));
    float da=max(r.x,r.y), db=max(r.y,r.z), dc=max(r.z,r.x);
    float c=(min(da,min(db,dc))-1.0)/s;
    d=max(d,c);
  }
  return d;
}
vec3 sceneMenger(vec2 uv, float t){
  vec2 p=(uv-0.5)*vec2(uRes.x/uRes.y,1.0)*1.6;
  float zoom=2.9+0.9*sin(t*0.08);
  vec3 ro=vec3(0.,0.,zoom), rd=normalize(vec3(p,-1.8));
  float dist=0.0; float d; float steps=0.0;
  for(int i=0;i<80;i++){
    d=mapMenger(ro+rd*dist,t);
    if(d<0.0015||dist>8.0) break;
    dist+=d*0.9; steps+=1.0;
  }
  vec3 bg=vec3(0.02,0.01,0.05)+vec3(0.20,0.04,0.16)*exp(-length(p)*1.5);
  if(dist>8.0) return bg;
  float ao=1.0-steps/80.0;
  float glow=steps/80.0;
  vec3 col=magmaPal(0.25+0.55*ao);
  col+=vec3(1.0,0.45,0.12)*pow(glow,2.0)*0.8;
  float fog=exp(-dist*0.28);
  return mix(bg,col,fog);
}

// ---------- 2 · FLUX : domain warping pleine resolution ----------
vec3 sceneFlux(vec2 uv, float t){
  vec2 p=uv*vec2(uRes.x/uRes.y,1.0)*2.6;
  vec2 m=(uMouse-0.5)*1.6;
  float md=length(uv-uMouse);
  vec2 q=vec2(fbm(p+vec2(t*0.09,0.)), fbm(p+vec2(5.2,1.3)-t*0.06));
  vec2 r=vec2(fbm(p+3.2*q+vec2(1.7,9.2)+m*0.8), fbm(p+3.2*q+vec2(8.3,2.8)-m*0.6));
  r+=0.35*exp(-md*3.5)*normalize(uv-uMouse+1e-4);
  float f=fbm(p+3.5*r);
  vec3 col=magmaPal(pow(f,1.25)+0.18*length(r)-0.1);
  col*=0.55+0.45*smoothstep(0.15,0.85,f);
  col+=vec3(1.0,0.55,0.2)*exp(-md*6.0)*0.25;
  return col;
}

// noyau commun des declinaisons de flux : deux niveaux de warp + souris
vec2 fluxField(vec2 p, vec2 uv, float t, out vec2 q, out float md){
  vec2 m=(uMouse-0.5)*1.6;
  md=length(uv-uMouse);
  q=vec2(fbm(p+vec2(t*0.09,0.)), fbm(p+vec2(5.2,1.3)-t*0.06));
  vec2 r=vec2(fbm(p+3.2*q+vec2(1.7,9.2)+m*0.8), fbm(p+3.2*q+vec2(8.3,2.8)-m*0.6));
  r+=0.35*exp(-md*3.5)*normalize(uv-uMouse+1e-4);
  return r;
}

// ---------- 4 · ENCRE : marbrure sumi-e + trainee magma pixelisee ----------
float trailAmount(vec2 uvq){
  // intensite cumulee des points estampilles (fade 1.2 s), distances en px ecran
  float g=0.0;
  for(int i=0;i<MAX_TRAIL;i++){
    if(i<uTrailCount){
      float age=uNow-uTrailStart[i];
      float timeF=clamp(1.0-age/1.2,0.,1.);
      float distPx=length((uvq-uTrailPos[i])*uRes);
      float rad=uTrailRad*min(uRes.x,uRes.y);
      float spatial=smoothstep(rad,rad*0.15,distPx);
      g=min(1.0, g+spatial*timeF*timeF);
    }
  }
  return g;
}
vec3 sceneInk(vec2 uv, float t){
  vec2 p=uv*vec2(uRes.x/uRes.y,1.0)*3.0+uSeed*vec2(17.31,9.77);
  vec2 q; float md;
  vec2 r=fluxField(p,uv,t*0.6,q,md);

  // trainee : echantillonnee au centre de gros blocs -> effet pixel
  float px=14.0*(uRes.x/1600.0+1.0);
  vec2 block=floor(uv*uRes/px);
  vec2 uvq=(block+0.5)*px/uRes;

  // bord liquide : le contour de la goutte ondule (bruit anime sur la position)
  float ar2=uRes.x/uRes.y;
  vec2 wob=vec2(fbm(uvq*vec2(ar2,1.0)*5.5+uNow*0.55),
                fbm(uvq*vec2(ar2,1.0)*5.5-uNow*0.45+7.7))-0.5;
  vec2 uvqW=uvq+wob*0.06;
  float g=trailAmount(uvqW);
  // gradient de la goutte -> normale du liquide (pour refraction + menisque)
  vec2 eps=vec2(px,0.)/uRes;
  float gx=trailAmount(uvqW+vec2(eps.x,0.));
  float gy=trailAmount(uvqW+vec2(0.,eps.x));
  vec2 grad=vec2(gx-g, gy-g);

  // la chaleur etire l'encre dans le sens du deplacement (sillage),
  // et la goutte refracte la marbrure comme une lentille d'eau
  vec2 dir=uTrailDir;
  vec2 perp=vec2(-dir.y,dir.x);
  vec2 smear=dir*2.8 + perp*0.5*sin(t*4.0+block.x*0.8+block.y*1.1);
  vec2 refr=grad*10.0 + wob*0.9*g;
  float fHot=fbm(p+3.8*r - g*smear + refr);
  float f=fbm(p+3.8*r);
  f=mix(f,fHot,min(1.0,g*1.2));

  // seuils nets superposes = voiles d'encre (plus rares, plus sombres)
  float ink=smoothstep(0.52,0.62,f);
  ink+=0.45*smoothstep(0.66,0.67,f);
  ink+=0.20*smoothstep(0.40,0.41,f)*(1.0-smoothstep(0.44,0.45,f));
  ink=clamp(ink,0.,1.);
  vec3 paper=vec3(0.006,0.006,0.006)+0.006*fbm(p*9.0);
  vec3 white=vec3(0.88,0.86,0.82);
  vec3 col=mix(paper, white, ink*(0.4+0.6*smoothstep(0.5,0.95,f)));
  col+=vec3(0.5,0.49,0.46)*exp(-md*6.0)*0.05;

  // fusion magma : la chaleur s'infiltre dans les veines d'encre,
  // le papier noir reste noir — seule la matiere s'embrase
  float dith=hash(block*1.7+3.1);
  float m=smoothstep(dith*0.5, dith*0.5+0.45, g);
  float flick=0.88+0.12*sin(uNow*7.0+dith*40.0);
  // presence de matiere : les veines brulent fort, le papier ne prend qu'au coeur
  float matter=smoothstep(0.05,0.75,ink)*0.95 + pow(g,1.6)*0.55;
  vec3 lava=magmaPal(clamp(0.10+ink*0.75+g*0.35,0.,1.))*flick;
  lava+=vec3(1.0,0.85,0.4)*pow(g,3.0)*(0.3+ink*0.6);
  // glyphes ASCII dans la braise : caractere choisi par la luminance,
  // qui "monte" vers les glyphes denses au coeur de la chaleur
  float glyph=1.0;
  if(uHasAtlas==1){
    vec2 cellUV=fract(uv*uRes/px);
    float lum=clamp(0.10+ink*0.62+g*0.30+0.08*sin(uNow*5.0+dith*30.0),0.,1.);
    float gi=floor(lum*(uGlyphCount-1.0)+0.5);
    float ax=mod(gi,uAtlasGrid.x), ay=floor(gi/uAtlasGrid.x);
    glyph=texture2D(uAtlas,(vec2(ax,ay)+cellUV)/uAtlasGrid).r;
  }
  float w=clamp(m*matter,0.,1.);
  col=mix(col, lava*(0.30+0.25*glyph), w);      // fond de braise tamise
  col+=lava*glyph*w*1.15;                        // le caractere brule par-dessus
  // menisque : liseret brillant sur le bord de la goutte, comme une tension de surface
  float rim=length(grad)*smoothstep(0.02,0.4,g)*(1.0-smoothstep(0.85,1.0,g));
  col+=vec3(1.0,0.72,0.35)*rim*5.5;
  // titre en masque dans le champ : les lettres se liquefient avec uMelt
  if(uHasTitle==1){
    vec2 tuv=vec2((uv.x-uTitleRect.x)/uTitleRect.z,(uv.y-uTitleRect.y)/uTitleRect.w);
    vec2 spill=(r-0.5)*0.9*uMelt;
    spill.y-=0.45*uMelt*fbm(tuv*5.0+uNow*0.35);
    vec2 duv=tuv+spill;
    if(duv.x>0.0&&duv.x<1.0&&duv.y>0.0&&duv.y<1.0){
      float a=texture2D(uTitle,vec2(duv.x,1.0-duv.y)).r;
      a=smoothstep(0.25,0.6,a)*(1.0-0.30*uMelt);
      col=mix(col,vec3(0.95,0.93,0.90),clamp(a,0.,1.));
      col+=magmaPal(0.6)*a*uMelt*0.5; // les lettres chauffent en fondant
    }
  }
  return col;
}

// ---------- 8 · DESCENTE : loss surface + bille de gradient ----------
vec3 sceneDescent(vec2 uv, float t){
  float ar=uRes.x/uRes.y;
  vec2 p=uv*vec2(ar,1.0)*2.4+uSeed*vec2(17.31,9.77);
  float h=fbm(p);
  // topographie de la loss surface (statique : la bille peut y descendre)
  float iso=abs(fract(h*22.0)-0.5);
  float line=smoothstep(0.13,0.03,iso);
  float major=step(0.8,fract(h*22.0/5.0+0.1));
  vec3 col=magmaPal(0.05+h*0.22)*0.30;
  col+=mix(vec3(1.0,0.62,0.22),vec3(1.0,0.90,0.70),major)*line*(0.28+0.55*major);
  // trace de la descente (stamps)
  float g=trailAmount(uv);
  col+=magmaPal(0.45+g*0.45)*g*0.8;
  // la bille (uMouse = position simulee cote JS)
  float d=length((uv-uMouse)*vec2(ar,1.0));
  col+=vec3(1.0,0.55,0.18)*exp(-d*d*700.0)*0.8;
  col+=vec3(1.0,0.92,0.65)*exp(-d*d*9000.0)*1.6;
  return col;
}

// ---------- 5 · TOPOGRAPHIE : courbes de niveau d'un terrain vivant ----------
vec3 sceneTopo(vec2 uv, float t){
  vec2 p=uv*vec2(uRes.x/uRes.y,1.0)*2.4;
  vec2 q; float md;
  vec2 r=fluxField(p,uv,t*0.5,q,md);
  float h=fbm(p+3.5*r);
  float iso=abs(fract(h*16.0)-0.5);
  float line=smoothstep(0.10,0.02,iso);
  float major=step(0.8,fract(h*16.0/5.0+0.1));
  float idx=floor(h*16.0);
  vec3 col=magmaPal(0.10+h*0.25)*0.35; // remplissage sombre par altitude
  vec3 lineCol=mix(vec3(1.0,0.72,0.28), vec3(1.0,0.92,0.75), major);
  col+=lineCol*line*(0.35+0.65*major)*(0.6+0.4*sin(idx*1.7));
  col+=vec3(1.0,0.6,0.2)*exp(-md*5.0)*0.2;
  return col;
}

// ---------- 6 · SOIE : filaments anisotropes irises ----------
vec3 sceneSilk(vec2 uv, float t){
  vec2 p=uv*vec2(uRes.x/uRes.y,1.0)*2.2;
  vec2 q; float md;
  vec2 r=fluxField(p,uv,t*0.7,q,md);
  float f=fbm(p+3.5*r);
  float threads=pow(abs(sin(f*34.0+q.x*7.0-t*0.5)),6.0);
  float sheen=pow(abs(sin(f*34.0+q.x*7.0-t*0.5+1.57)),40.0);
  vec3 tintA=vec3(0.15,0.45,0.85), tintB=vec3(0.85,0.25,0.75), tintC=vec3(1.0,0.85,0.55);
  vec3 col=vec3(0.015,0.01,0.03);
  col+=mix(tintA,tintB,q.y)*threads*0.75;
  col+=tintC*sheen*0.9;
  col+=mix(tintA,tintB,0.5)*exp(-md*5.0)*0.25;
  return col;
}

// ---------- 7 · ABYSSE : bioluminescence des profondeurs ----------
vec3 sceneAbyss(vec2 uv, float t){
  vec2 p=uv*vec2(uRes.x/uRes.y,1.0)*2.8;
  vec2 q; float md;
  vec2 r=fluxField(p,uv,t*0.45,q,md);
  float f=fbm(p+3.5*r);
  vec3 col=mix(vec3(0.005,0.015,0.045), vec3(0.02,0.10,0.16), uv.y+0.3*q.y);
  vec3 cyan=vec3(0.10,0.95,0.85), blue=vec3(0.15,0.45,1.0);
  float veil=pow(smoothstep(0.35,0.85,f),2.0);
  col+=mix(blue,cyan,q.x)*veil*0.65;
  // plancton : points scintillants sur les cretes du champ
  vec2 cell=floor(uv*uRes/9.0);
  float sparkle=step(0.985,hash(cell))*pow(0.5+0.5*sin(t*3.0+hash(cell.yx)*40.0),3.0);
  col+=cyan*sparkle*smoothstep(0.4,0.8,f)*1.2;
  col+=cyan*exp(-md*4.5)*0.30*(0.6+0.4*sin(t*2.0));
  return col;
}

// ---------- 3 · CAUSTIQUES : eau incandescente ----------
vec3 sceneCaustics(vec2 uv, float t){
  vec2 p=mod((uv+(uMouse-0.5)*0.4)*vec2(uRes.x/uRes.y,1.0)*6.2831, 6.2831)-250.0;
  vec2 i=p;
  float c=1.0, inten=0.005;
  for(int n=0;n<5;n++){
    float t2=t*0.35*(1.0-(3.5/float(n+1)));
    i=p+vec2(cos(t2-i.x)+sin(t2+i.y), sin(t2-i.y)+cos(t2+i.x));
    c+=1.0/length(vec2(p.x/(sin(i.x+t2)/inten), p.y/(cos(i.y+t2)/inten)));
  }
  c/=5.0;
  c=1.17-pow(max(c,0.0),1.4);
  float v=pow(abs(c),8.0);
  vec3 col=magmaPal(clamp(0.12+v*0.95,0.,1.))*(0.25+v*1.1);
  return col;
}

// ---------- 9 · RESEAU : un vrai reseau de neurones, forward pass anime ----------
float sdSeg(vec2 p, vec2 a, vec2 b){
  vec2 pa=p-a, ba=b-a;
  float h=clamp(dot(pa,ba)/dot(ba,ba),0.,1.);
  return length(pa-ba*h);
}
float layerN(float L){ return L<0.5?5.0:(L>2.5?4.0:7.0); }
// y de la fibre k a l'abscisse x : les fibres convergent dans les noeuds
// aux "portes" de chaque couche puis se redeploient — faisceaux organiques
float threadY(float k, float x, float t){
  float seg=clamp((x-0.10)/(0.80/3.0),0.0,2.999);
  float L=floor(seg);
  float u=seg-L;
  float nA=layerN(L), nB=layerN(L+1.0);
  float iA=floor(hash(vec2(k*1.7,L*9.3))*nA);
  float iB=floor(hash(vec2(k*1.7,(L+1.0)*9.3))*nB);
  float yA=mix(0.24,0.76,(iA+0.5)/nA);
  float yB=mix(0.24,0.76,(iB+0.5)/nB);
  float e=u*u*u*(u*(u*6.0-15.0)+10.0); // quintic : tangentes plates aux noeuds
  float y=mix(yA,yB,e);
  // ondulation nulle aux noeuds, maximale entre deux
  y+=0.020*(noise(vec2(x*4.0+k*13.1, t*0.10+k*0.7))-0.5)*2.0*sin(3.1416*u);
  return y;
}
vec3 sceneNetwork(vec2 uv, float t){
  float ar=uRes.x/uRes.y;
  vec3 col=vec3(0.0);
  vec3 ivory=vec3(0.92,0.89,0.83);
  vec3 gold=vec3(1.0,0.72,0.30);
  float px1=1.0/uRes.y;
  // 40 fibres accumulees en additif = longue exposition
  for(int ki=0;ki<40;ki++){
    float k=float(ki);
    float z=hash(vec2(k,99.1)); // profondeur : 0 loin, 1 proche
    // parallaxe : la souris deplace chaque plan selon sa profondeur
    vec2 par=(uMouse-0.5)*vec2(0.030,0.022)*(z-0.5);
    float x=uv.x+par.x;
    float y=threadY(k,x,t)+par.y;
    float dy=uv.y-y;
    // profondeur de champ : les fibres proches sont larges et douces,
    // les lointaines fines et seches
    float sig=px1*(0.6+3.2*z*z);
    float line=exp(-dy*dy/(2.0*sig*sig));
    float lum=0.014+0.050*z;
    // legere teinte : les plans lointains tirent vers le froid
    vec3 tint=mix(ivory*vec3(0.75,0.82,1.0),ivory,z);
    col+=tint*line*lum;
    // perles en transit (2 par fibre), easing et vitesse propres
    for(int b=0;b<2;b++){
      float ph=hash(vec2(k,float(b)*3.7+1.0));
      float sp=0.035+0.045*hash(vec2(k,float(b)+7.7));
      float s=fract(t*sp+ph);
      float e=s*s*(3.0-2.0*s);
      float xb=mix(0.04,0.96,e);
      float yb=threadY(k,xb+par.x,t)+par.y;
      vec2 d2=vec2((uv.x-xb)*ar,(uv.y-yb));
      float bd=dot(d2,d2);
      float bsig=px1*(1.4+3.5*z);
      float amp=0.25+0.75*z;
      col+=gold*exp(-bd/(2.0*bsig*bsig))*amp*0.9;
      col+=gold*exp(-bd/(2.0*bsig*bsig*40.0))*amp*0.10;
      // sillage longue exposition derriere la perle, sur la fibre
      float behind=exp(-max(0.0,xb-uv.x)*22.0)*step(uv.x,xb);
      col+=gold*line*behind*amp*0.30;
    }
  }
  // les noeuds emergent de la convergence ; on ajoute juste un souffle de lumiere
  for(int L=0;L<4;L++){
    float fL=float(L);
    float n=layerN(fL);
    float gx=0.10+0.80*fL/3.0;
    for(int i=0;i<7;i++){
      if(float(i)>=n) continue;
      float yn=mix(0.24,0.76,(float(i)+0.5)/n);
      vec2 d2=vec2((uv.x-gx)*ar,(uv.y-yn));
      float dd=dot(d2,d2);
      col+=vec3(1.0,0.96,0.88)*exp(-dd*7000.0)*0.10;
      col+=vec3(0.9,0.75,0.5)*exp(-dd*600.0)*0.020;
    }
  }
  // voile atmospherique diagonal, tres discret
  col+=vec3(0.020,0.016,0.012)*smoothstep(0.2,1.4,1.0-uv.y+uv.x*0.3)*0.5;
  col*=smoothstep(1.9,0.6,length(uv-0.5)*1.6);
  return col;
}

// ---------- 10 · CORTEX : le reseau en volume (3D projete, bokeh, fog) ----------
vec3 nodePos3(float L, float i, float n){
  float x=mix(-1.45,1.45,L/3.0)+(hash(vec2(L*7.1,i*3.3))-0.5)*0.30;
  float th=(i+0.5)/n*6.2831+L*1.7;
  float rad=0.28+0.55*hash(vec2(i*9.7,L*2.9));
  float y=sin(th)*rad*0.75+(hash(vec2(i*5.1,L*11.3))-0.5)*0.25;
  float z=cos(th)*rad+(hash(vec2(i*2.7,L*5.9))-0.5)*0.25;
  return vec3(x,y,z);
}
vec2 project3(vec3 P, float ca, float sa, float cb, float sb, out float depth){
  P.xz=mat2(ca,-sa,sa,ca)*P.xz;            // orbite (souris + temps)
  P.yz=mat2(cb,-sb,sb,cb)*P.yz;            // elevation
  P.z+=3.1;
  depth=P.z;
  return P.xy*(2.35/P.z);
}
vec3 sceneCortex(vec2 uv, float t){
  float ar=uRes.x/uRes.y;
  vec2 q=(uv-0.5)*vec2(ar,1.0)*2.0;
  vec3 col=vec3(0.0);
  vec3 ivory=vec3(0.92,0.89,0.83);
  vec3 gold=vec3(1.0,0.72,0.30);
  float px1=2.0/uRes.y;
  float a=t*0.10+(uMouse.x-0.5)*1.4;
  float b=0.22+(uMouse.y-0.5)*0.7;
  float ca=cos(a), sa=sin(a), cb=cos(b), sb=sin(b);
  float focus=3.1;
  // fils : hairlines 3D projetees, fog + profondeur de champ
  for(int L=0;L<3;L++){
    float fL=float(L);
    float nA=layerN(fL), nB=layerN(fL+1.0);
    for(int i=0;i<7;i++){
      if(float(i)>=nA) continue;
      float dA;
      vec2 A=project3(nodePos3(fL,float(i),nA),ca,sa,cb,sb,dA);
      for(int j=0;j<7;j++){
        if(float(j)>=nB) continue;
        float w=hash(vec2(fL*31.0+float(i)*7.0,float(j)*13.0));
        if(w<0.55) continue;
        float dB;
        vec2 B=project3(nodePos3(fL+1.0,float(j),nB),ca,sa,cb,sb,dB);
        float d=sdSeg(q,A,B);
        float depth=0.5*(dA+dB);
        float blur=abs(depth-focus);
        float sig=px1*(0.6+blur*5.0);
        float line=exp(-d*d/(2.0*sig*sig));
        float fog=exp(-max(0.0,depth-2.4)*0.55);
        col+=ivory*line*(0.030+0.055*w)*fog;
        // perle en transit sur certains fils
        if(w>0.87){
          float s=fract(t*(0.05+0.06*hash(vec2(w,fL)))+w*7.0);
          float e=s*s*(3.0-2.0*s);
          float dP;
          vec2 P=project3(mix(nodePos3(fL,float(i),nA),nodePos3(fL+1.0,float(j),nB),e),ca,sa,cb,sb,dP);
          vec2 dv=q-P;
          float bsig=px1*(1.4+abs(dP-focus)*6.0+ (3.1/dP)*0.9);
          float bd=dot(dv,dv);
          float bfog=exp(-max(0.0,dP-2.4)*0.55);
          col+=gold*exp(-bd/(2.0*bsig*bsig))*bfog*1.3;
          col+=gold*exp(-bd/(2.0*bsig*bsig*30.0))*bfog*0.12;
        }
      }
    }
  }
  // noeuds : disques bokeh — nets au plan focal, larges et doux ailleurs
  for(int L=0;L<4;L++){
    float fL=float(L);
    float n=layerN(fL);
    for(int i=0;i<7;i++){
      if(float(i)>=n) continue;
      float dN;
      vec2 N=project3(nodePos3(fL,float(i),n),ca,sa,cb,sb,dN);
      float dd=length(q-N);
      float blur=abs(dN-focus);
      float r=px1*(2.6+blur*10.0)*(3.1/dN);
      float disc=smoothstep(r,r*0.35,dd);
      float edge=smoothstep(r,r*0.85,dd)-smoothstep(r*0.85,r*0.6,dd); // anneau bokeh
      float fog=exp(-max(0.0,dN-2.4)*0.6);
      float tw=0.7+0.3*sin(t*1.3+float(i)*2.7+fL*4.1);
      col+=ivory*disc*(0.16+0.30/max(0.6,blur*3.0))*fog*tw;
      col+=ivory*edge*0.10*fog;
    }
  }
  // poussiere en suspension, tres discrete
  vec2 cell=floor(uv*uRes/13.0);
  float dust=step(0.995,hash(cell*1.3));
  col+=ivory*dust*0.05*(0.5+0.5*sin(t*1.5+hash(cell.yx)*40.0));
  col*=smoothstep(2.0,0.62,length(uv-0.5)*1.6);
  return col;
}

// ---------- 11 · CERVEAU : silhouette abstraite, circonvolutions irisees ----------
vec3 prismPal(float h){
  return 0.5+0.5*cos(6.2831*(h+vec3(0.00,0.33,0.67)));
}
float sdEll(vec2 p, vec2 c, vec2 r){ return length((p-c)/r)-1.0; }
float sminf(float a,float b,float k){ float h=clamp(0.5+0.5*(b-a)/k,0.,1.); return mix(b,a,h)-k*h*(1.0-h); }
// profil de cerveau (vue laterale) : lobes + cervelet + tronc
float brainSDF(vec2 p, out float cere){
  float d=sdEll(p,vec2(0.00,0.12),vec2(0.50,0.36));           // cerebrum
  d=sminf(d, sdEll(p,vec2(-0.34,0.02),vec2(0.24,0.27)),0.10); // lobe frontal
  d=sminf(d, sdEll(p,vec2(0.34,0.06),vec2(0.24,0.24)),0.10);  // lobe occipital
  d=sminf(d, sdEll(p,vec2(-0.04,-0.12),vec2(0.36,0.20)),0.08);// lobe temporal
  cere=sdEll(p,vec2(0.33,-0.27),vec2(0.17,0.12));             // cervelet
  d=sminf(d,cere,0.035);
  d=sminf(d, sdEll(p,vec2(0.14,-0.36),vec2(0.06,0.11)),0.05); // tronc
  return d;
}
vec3 sceneBrainArt(vec2 uv, float t){
  float ar=uRes.x/uRes.y;
  vec2 p=(uv-vec2(0.5,0.52))*vec2(ar,1.0)*1.25;
  p+=(uMouse-0.5)*vec2(0.05,0.04); // legere parallaxe
  float cere;
  float d=brainSDF(p,cere);
  float inside=smoothstep(0.012,-0.012,d);
  float inCere=smoothstep(0.012,-0.012,cere);
  vec3 col=vec3(0.008,0.008,0.016);
  float ang=atan(p.y,p.x)/6.2831;
  // palette resserree : indigo -> bleu electrique -> cyan -> rose
  // (definie inline pour rester locale a la scene)
  float w=fbm(p*2.3+vec2(t*0.030,-t*0.022));
  float w2=fbm(p*4.6+vec2(-t*0.018,t*0.026)+7.7);
  float field=(p.y+0.85*p.x+2.3*w)*5.2+1.5*w2;
  float iso=abs(fract(field)-0.5);
  float line=smoothstep(0.16,0.02,iso);            // sillons principaux, fins
  float iso2=abs(fract(field*2.7+w2*2.0)-0.5);
  float line2=smoothstep(0.14,0.02,iso2)*0.35;     // reseau secondaire discret
  float hx=clamp(0.15+w*0.9+p.x*0.25,0.,1.);
  vec3 cIndigo=vec3(0.16,0.10,0.50), cBlue=vec3(0.15,0.45,1.00),
       cCyan=vec3(0.30,0.95,0.95), cRose=vec3(1.00,0.38,0.62);
  vec3 hue = hx<0.45 ? mix(cIndigo,cBlue,hx/0.45)
           : (hx<0.8 ? mix(cBlue,cCyan,(hx-0.45)/0.35)
                     : mix(cCyan,cRose,(hx-0.8)/0.2));
  // cervelet : stries fines paralleles
  float striaF=(p.y+p.x*0.55+0.7*fbm(p*5.0))*16.0;
  float stria=smoothstep(0.18,0.03,abs(fract(striaF)-0.5));
  float body=inside*(1.0-inCere);
  // onde de pensee : parcourt le cortex, discrete
  float wave=0.45+0.75*exp(-abs(fract(t*0.08)*3.0-1.5-p.x)*3.5);
  // matiere de verre : presque rien, tout est dans les filaments
  col+=hue*body*0.045;
  col+=hue*line*body*0.85*wave;
  col+=mix(cIndigo,cBlue,0.5)*line2*body*wave;
  col+=mix(cBlue,cCyan,0.5)*stria*inCere*inside*0.55;
  col+=cBlue*inCere*inside*0.04;
  // synapses : etincelles rares qui pulsent sur les sillons
  vec2 cellS=floor(p*26.0);
  float sp=hash(cellS*1.31);
  if(sp>0.975){
    vec2 sc=(cellS+0.5)/26.0;
    float sd2=length(p-sc);
    float tw2=pow(0.5+0.5*sin(t*2.2+sp*40.0),3.0);
    col+=mix(cCyan,vec3(1.0),0.5)*exp(-sd2*sd2*3000.0)*tw2*body*1.2;
  }
  // bord : liseret net + halo discret
  float rim=exp(-abs(d)*140.0);
  col+=mix(cBlue,cCyan,0.5+0.5*sin(ang*6.2831+t*0.1))*rim*0.40;
  col+=cIndigo*exp(-max(0.0,d)*5.0)*0.10;
  col=col/(1.0+col*0.45);
  col*=0.94+0.06*sin(uv.y*uRes.y*0.8);
  // ---- goutte de glyphes (fusion, comme Encre) ----
  float px=14.0*(uRes.x/1600.0+1.0);
  vec2 block=floor(uv*uRes/px);
  vec2 uvq=(block+0.5)*px/uRes;
  vec2 wob=vec2(fbm(uvq*vec2(ar,1.0)*5.5+uNow*0.55),
                fbm(uvq*vec2(ar,1.0)*5.5-uNow*0.45+7.7))-0.5;
  float g=trailAmount(uvq+wob*0.05);
  if(g>0.003){
    float dith=hash(block*1.7+3.1);
    float m=smoothstep(dith*0.5, dith*0.5+0.45, g);
    float lum2=clamp(dot(col,vec3(0.35))+g*0.35,0.,1.);
    float glyph=1.0;
    if(uHasAtlas==1){
      vec2 cellUV=fract(uv*uRes/px);
      float gi=floor(clamp(lum2+0.08*sin(uNow*5.0+dith*30.0),0.,1.)*(uGlyphCount-1.0)+0.5);
      float ax=mod(gi,uAtlasGrid.x), ay=floor(gi/uAtlasGrid.x);
      glyph=texture2D(uAtlas,(vec2(ax,ay)+cellUV)/uAtlasGrid).r;
    }
    vec3 gh=prismPal(lum2*0.7+uNow*0.03+dith*0.15);
    gh=mix(vec3(dot(gh,vec3(0.333))),gh,1.8);
    col=mix(col, col*0.45, m*inside); // ne mord que sur la matiere
    col+=gh*glyph*m*2.2*(0.3+0.7*inside);
    col+=vec3(1.0)*glyph*m*(0.30+pow(g,2.0)*0.8)*inside;
  }
  col*=smoothstep(2.0,0.65,length(uv-0.5)*1.6);
  return col;
}

void main(){
  vec2 uv=vUV;
  vec3 col;
  if(uScene==1) col=sceneMenger(uv,uTime);
  else if(uScene==2) col=sceneFlux(uv,uTime);
  else if(uScene==3) col=sceneCaustics(uv,uTime);
  else if(uScene==4) col=sceneInk(uv,uTime);
  else if(uScene==5) col=sceneTopo(uv,uTime);
  else if(uScene==6) col=sceneSilk(uv,uTime);
  else if(uScene==7) col=sceneAbyss(uv,uTime);
  else if(uScene==8) col=sceneDescent(uv,uTime);
  else if(uScene==9) col=sceneNetwork(uv,uTime);
  else if(uScene==10) col=sceneCortex(uv,uTime);
  else if(uScene==11) col=sceneBrainArt(uv,uTime);
  else col=sceneLiquid(uv,uTime);
  // grain leger + vignette
  col*=0.97+0.06*hash(uv*uRes+fract(uTime)*7.0);
  col*=smoothstep(1.55,0.45,length(uv-0.5)*1.5);
  gl_FragColor=vec4(col,1.0);
}`;

const raws: RawRenderer[] = [];
let rawLoop = false;
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// fbm JS identique au fbm GLSL (pour que la bille descende la surface rendue)
function jsFract(x: number) { return x - Math.floor(x); }
function jsHash(x: number, y: number) { return jsFract(Math.sin(x * 127.1 + y * 311.7) * 43758.5453); }
function jsNoise(x: number, y: number) {
  const xi = Math.floor(x), yi = Math.floor(y);
  let fx = x - xi, fy = y - yi;
  fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
  return (
    jsHash(xi, yi) * (1 - fx) * (1 - fy) + jsHash(xi + 1, yi) * fx * (1 - fy) +
    jsHash(xi, yi + 1) * (1 - fx) * fy + jsHash(xi + 1, yi + 1) * fx * fy
  );
}
function jsFbm(x: number, y: number) {
  let v = 0, a = 0.5;
  for (let k = 0; k < 5; k++) { v += a * jsNoise(x, y); x = x * 2.03 + 11.7; y = y * 2.03 + 11.7; a *= 0.5; }
  return v;
}

class RawRenderer {
  wrap: HTMLElement; canvas: HTMLCanvasElement;
  gl!: WebGLRenderingContext; prog!: WebGLProgram;
  u: Record<string, WebGLUniformLocation | null> = {};
  mouse = { x: 0.5, y: 0.5 }; smooth = { x: 0.5, y: 0.5 };
  dpr: number;
  trail: { x: number; y: number; tSec: number }[] = [];
  lastStamp = 0;
  dir = { x: 0, y: 0 };
  trailRadValue = 0.11;
  private trailPos = new Float32Array(32);
  private trailStart = new Float32Array(16);

  scene: number;
  seed: number;
  melt = 0; meltTarget = 0;
  descent: { x: number; y: number; vx: number; vy: number; holdUntil: number } | null = null;

  constructor(wrap: HTMLElement, scene: number, seed = 0, trailRad = 0.11) {
    this.wrap = wrap;
    this.scene = scene;
    this.seed = seed;
    this.trailRadValue = trailRad;
    this.canvas = wrap.querySelector("canvas")!;
    this.dpr = Math.max(1, Math.min(2, devicePixelRatio || 1));
    const gl = this.canvas.getContext("webgl", { alpha: false, antialias: false });
    if (!gl) return;
    this.gl = gl;
    const mk = (t: number, s: string) => {
      const sh = gl.createShader(t)!; gl.shaderSource(sh, s); gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(sh));
      return sh;
    };
    const p = this.prog = gl.createProgram()!;
    gl.attachShader(p, mk(gl.VERTEX_SHADER, VS));
    gl.attachShader(p, mk(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p); gl.useProgram(p);
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    const aPos = gl.getAttribLocation(p, "aPos");
    gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
    for (const name of ["uRes", "uTime", "uScene", "uMouse", "uTrailCount", "uTrailPos", "uTrailStart", "uNow", "uTrailDir",
      "uAtlas", "uAtlasGrid", "uGlyphCount", "uHasAtlas", "uSeed", "uTrailRad", "uTitle", "uTitleRect", "uMelt", "uHasTitle"])
      this.u[name] = gl.getUniformLocation(p, name);
    gl.uniform1i(this.u.uScene, scene);
    gl.uniform1i(this.u.uTrailCount, 0);
    gl.uniform1i(this.u.uHasAtlas, 0);
    gl.uniform1i(this.u.uHasTitle, 0);
    gl.uniform1f(this.u.uMelt, 0);
    gl.uniform1f(this.u.uSeed, seed);
    gl.uniform1f(this.u.uTrailRad, trailRad);
    if (scene === 8) this.descent = { x: 0.2 + 0.6 * Math.random(), y: 0.75, vx: 0, vy: 0, holdUntil: 0 };
    // atlas de glyphes GeistMono partage avec le renderer ASCII (chiffres + symboles)
    buildAtlas().then((atlas) => {
      gl.useProgram(this.prog);
      const tex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, gl.LUMINANCE, gl.UNSIGNED_BYTE, atlas.canvas);
      gl.uniform1i(this.u.uAtlas, 1);
      gl.uniform2f(this.u.uAtlasGrid, atlas.gridCols, atlas.gridRows);
      gl.uniform1f(this.u.uGlyphCount, DENSITY.length);
      gl.uniform1i(this.u.uHasAtlas, 1);
      gl.activeTexture(gl.TEXTURE0);
    });

    new ResizeObserver(() => this.resize()).observe(wrap);
    this.resize();
    raws.push(this);
    requestAnimationFrame(() => wrap.classList.add("active"));
  }

  resize() {
    const r = this.wrap.getBoundingClientRect();
    const w = Math.max(2, Math.floor(r.width * this.dpr)), h = Math.max(2, Math.floor(r.height * this.dpr));
    if (this.canvas.width === w && this.canvas.height === h) return;
    this.canvas.width = w; this.canvas.height = h;
    this.gl.viewport(0, 0, w, h);
    this.gl.useProgram(this.prog);
    this.gl.uniform2f(this.u.uRes, w, h);
  }

  stamp(clientX: number, clientY: number, nowMs: number) {
    if (nowMs - this.lastStamp < 22) return;
    const r = this.wrap.getBoundingClientRect();
    if (clientX < r.left || clientX > r.right || clientY < r.top || clientY > r.bottom) return;
    this.lastStamp = nowMs;
    const pt = { x: (clientX - r.left) / r.width, y: 1 - (clientY - r.top) / r.height, tSec: nowMs / 1000 };
    const prev = this.trail[0];
    if (prev) {
      // direction lissee, amplitude bornee par la vitesse
      const dx = pt.x - prev.x, dy = pt.y - prev.y;
      const len = Math.hypot(dx, dy) + 1e-6;
      const speed = Math.min(1, len * 18);
      this.dir.x += (dx / len * speed - this.dir.x) * 0.25;
      this.dir.y += (dy / len * speed - this.dir.y) * 0.25;
    }
    this.trail.unshift(pt);
    if (this.trail.length > 16) this.trail.length = 16;
  }

  // un pas de descente de gradient avec momentum sur la surface jsFbm
  stepDescent(nowMs: number) {
    const d = this.descent!;
    if (nowMs < d.holdUntil) return;
    const ar = this.canvas.width / Math.max(1, this.canvas.height);
    const h = (x: number, y: number) => jsFbm(x * ar * 2.4 + this.seed * 17.31, y * 2.4 + this.seed * 9.77);
    const e = 0.004;
    const gx = (h(d.x + e, d.y) - h(d.x - e, d.y)) / (2 * e);
    const gy = (h(d.x, d.y + e) - h(d.x, d.y - e)) / (2 * e);
    d.vx = 0.90 * d.vx - 0.00016 * gx;
    d.vy = 0.90 * d.vy - 0.00016 * gy;
    const sp = Math.hypot(d.vx, d.vy);
    if (sp > 0.004) { d.vx *= 0.004 / sp; d.vy *= 0.004 / sp; }
    d.x += d.vx; d.y += d.vy;
    const out = d.x < 0.03 || d.x > 0.97 || d.y < 0.05 || d.y > 0.95;
    if ((sp < 0.00006 && Math.hypot(gx, gy) < 0.35) || out) {
      // convergence (minimum local) ou sortie : pause puis nouveau depart
      d.holdUntil = nowMs + 1500;
      d.x = 0.15 + 0.7 * Math.random(); d.y = 0.2 + 0.65 * Math.random();
      d.vx = 0; d.vy = 0;
    }
    this.mouse.x = d.x; this.mouse.y = d.y;
    this.smooth.x = d.x; this.smooth.y = d.y;
    const r = this.wrap.getBoundingClientRect();
    this.stamp(r.left + r.width * d.x, r.top + r.height * (1 - d.y), nowMs);
  }

  draw(t: number) {
    const gl = this.gl;
    if (!gl) return;
    this.melt += (this.meltTarget - this.melt) * 0.07;
    gl.useProgram(this.prog);
    gl.uniform1f(this.u.uMelt, this.melt);
    this.smooth.x += (this.mouse.x - this.smooth.x) * 0.06;
    this.smooth.y += (this.mouse.y - this.smooth.y) * 0.06;
    gl.useProgram(this.prog);
    this.trail = this.trail.filter((p) => t - p.tSec <= 1.4);
    const n = Math.min(16, this.trail.length);
    for (let i = 0; i < n; i++) {
      const p = this.trail[i];
      this.trailPos[i * 2] = p.x; this.trailPos[i * 2 + 1] = p.y;
      this.trailStart[i] = p.tSec;
    }
    gl.uniform1i(this.u.uTrailCount, n);
    if (n) {
      gl.uniform2fv(this.u.uTrailPos, this.trailPos);
      gl.uniform1fv(this.u.uTrailStart, this.trailStart);
    }
    // la direction decroit quand la souris s'arrete
    this.dir.x *= 0.96; this.dir.y *= 0.96;
    gl.uniform2f(this.u.uTrailDir, this.dir.x, this.dir.y);
    gl.uniform1f(this.u.uNow, t);
    gl.uniform1f(this.u.uTime, reduced() ? 15.0 : t);
    gl.uniform2f(this.u.uMouse, this.smooth.x, this.smooth.y);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}

const registry = new Map<HTMLElement, RawRenderer>();

// #3 — titre en masque dans le champ d'encre : rendu dans le shader, fond au hover
export async function attachTitleMelt(wrap: HTMLElement, title: HTMLElement) {
  const r = registry.get(wrap);
  if (!r || !r.gl) return;
  try { await document.fonts.ready; } catch {}
  const gl = r.gl;
  const draw = () => {
    const wr = wrap.getBoundingClientRect(), tr = title.getBoundingClientRect();
    const dpr = Math.min(2, devicePixelRatio || 1);
    const cv = document.createElement("canvas");
    cv.width = Math.max(2, Math.ceil(tr.width * dpr));
    cv.height = Math.max(2, Math.ceil(tr.height * dpr));
    const g = cv.getContext("2d")!;
    g.fillStyle = "#000"; g.fillRect(0, 0, cv.width, cv.height);
    g.fillStyle = "#fff";
    const cs = getComputedStyle(title);
    g.font = `${cs.fontWeight} ${parseFloat(cs.fontSize) * dpr}px ${cs.fontFamily}`;
    g.textBaseline = "top";
    const lh = parseFloat(cs.lineHeight) * dpr;
    title.innerText.split("\n").forEach((line, i) => g.fillText(line, 0, i * lh + lh * 0.08));
    gl.useProgram(r.prog);
    const tex = gl.createTexture();
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.LUMINANCE, gl.LUMINANCE, gl.UNSIGNED_BYTE, cv);
    gl.uniform1i(r.u.uTitle, 2);
    gl.uniform4f(r.u.uTitleRect,
      (tr.left - wr.left) / wr.width,
      1 - (tr.bottom - wr.top) / wr.height,
      tr.width / wr.width,
      tr.height / wr.height);
    gl.uniform1i(r.u.uHasTitle, 1);
    gl.activeTexture(gl.TEXTURE0);
    title.style.color = "transparent"; // le shader dessine le texte a sa place
  };
  draw();
  addEventListener("resize", draw, { passive: true });
  title.addEventListener("pointerenter", () => { r.meltTarget = 1; });
  title.addEventListener("pointerleave", () => { r.meltTarget = 0; });
}

export function initRaw(wrap: HTMLElement, scene: number, seed = 0, trailRad = 0.11) {
  registry.set(wrap, new RawRenderer(wrap, scene, seed, trailRad));
  if (!rawLoop) {
    rawLoop = true;
    let lastRealMove = -1e9; // idle au chargement : la goutte fantome demarre seule
    addEventListener("pointermove", (e) => {
      const now = performance.now();
      lastRealMove = now;
      for (const r of raws) {
        if (r.descent) continue; // la bille de gradient ignore la souris
        const rect = r.wrap.getBoundingClientRect();
        r.mouse.x = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
        r.mouse.y = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
        r.stamp(e.clientX, e.clientY, now);
      }
    }, { passive: true });
    const loop = (now: number) => {
      // mode idle : apres 2.5 s sans souris, une goutte fantome derive en lissajous
      if (now - lastRealMove > 2500 && !reduced()) {
        const t = now / 1000;
        const ax = 0.5 + 0.31 * Math.sin(t * 0.21) + 0.10 * Math.sin(t * 0.53);
        const ay = 0.48 + 0.26 * Math.cos(t * 0.17) + 0.09 * Math.cos(t * 0.47);
        for (const r of raws) {
          if (r.descent) continue;
          const rect = r.wrap.getBoundingClientRect();
          r.mouse.x = ax; r.mouse.y = ay;
          r.stamp(rect.left + rect.width * ax, rect.top + rect.height * ay, now);
        }
      }
      for (const r of raws) {
        if (r.descent && !reduced()) r.stepDescent(now);
        r.draw(now / 1000);
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
}
