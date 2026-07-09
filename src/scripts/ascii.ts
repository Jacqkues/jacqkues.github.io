// Port du renderer ASCII WebGL « ascii-105 » de generalintuition.com :
// fragment shader qui découpe l'écran en cellules, échantillonne une scène
// procédurale, mappe la luminance sur un atlas de glyphes GeistMono, et
// applique un glow au curseur avec traînée estampillée toutes les 25 ms.

export const DENSITY = String.raw`$@B%8&WM#*oahkbdpqwmZO0QLCJUYZXcvunxrj/ft\|()1{}[]?_-+~<>i!lI;:",^'. `;
const CFG = {
  cellSizePx: 10, cellAspectRatio: 0.6, charAspectRatio: 0.85, charFillRatio: 1,
  gamma: 1.6, tileOpacity: 0.5, glyphOpacity: 1.0, blendMode: 1, blendStrength: 1, preserveHue: 1,
  glowRadiusMultiplier: 8, glowDurationSec: 1, glowIntensity: 2.5, glowOpacity: 1,
  glowBlendMode: 0, glowUseTileColor: 1, glowSaturationBoost: 1, glowInnerFrac: 1,
  glowFalloffExp: 0.1, glowLumaGain: 0.6, glowLumaExp: 1.5, glowShrinkStrength: 0.2,
  glowStampIntervalMs: 25, dprCap: 2, atlasGlyphSize: 50,
};

const VS = `attribute vec2 aPos; varying vec2 vUV;
void main(){ vUV = aPos*0.5+0.5; gl_Position = vec4(aPos,0.,1.); }`;

const FS = `
precision highp float;
varying vec2 vUV;
uniform sampler2D uAtlas;
uniform vec2 uCanvasSize; uniform float uCols; uniform float uRows; uniform vec2 uInvGrid;
uniform vec2 uCellSizePx; uniform float uCharAspectRatio; uniform float uCharFillRatio;
uniform float uGamma; uniform float uTileOpacity; uniform float uGlyphOpacity;
uniform int uBlendMode; uniform float uBlendStrength; uniform int uPreserveHue;
uniform float uGlyphCount; uniform vec2 uAtlasGrid; uniform float uTime; uniform int uScene;
uniform sampler2D uImage; uniform int uUseImage; uniform vec2 uCropMin; uniform vec2 uCropMax;
uniform vec2 uJuliaC;

const int MAX_GLOW = 8;
uniform int uGlowCount; uniform vec2 uGlowCenters[MAX_GLOW];
uniform float uGlowRadiiPx[MAX_GLOW]; uniform float uGlowStart[MAX_GLOW];
uniform float uNow; uniform float uGlowDuration; uniform float uGlowIntensity;
uniform float uGlowOpacity; uniform int uGlowBlendMode; uniform int uGlowUseTileColor;
uniform float uGlowSaturationBoost; uniform float uGlowInnerFrac; uniform float uGlowFalloffExp;
uniform float uGlowLumaGain; uniform float uGlowLumaExp; uniform float uGlowShrinkStrength;

float luminance(vec3 rgb){ return dot(rgb, vec3(0.299,0.587,0.114)); }
vec3 blendScreen(vec3 b, vec3 t){ return 1.0-(1.0-b)*(1.0-t); }
vec3 blendAdd(vec3 b, vec3 t){ return clamp(b+t,0.,1.); }
vec3 blendColorDodge(vec3 b, vec3 t){ vec3 d = max(vec3(0.001),1.0-t); return clamp(b/d,0.,1.); }
vec3 doBlend(vec3 b, vec3 t, int m){ if(m==1) return blendAdd(b,t); if(m==2) return blendColorDodge(b,t); if(m==3) return t; return blendScreen(b,t); }
vec3 retintToBase(vec3 c, vec3 base){ float bl=max(0.0001,luminance(base)); float cl=max(0.0001,luminance(c)); return clamp(base*(cl/bl),0.,1.); }
vec3 rgb2hsv(vec3 c){ float cMax=max(c.r,max(c.g,c.b)); float cMin=min(c.r,min(c.g,c.b)); float d=cMax-cMin; float h=0.;
  if(d>1e-5){ if(cMax==c.r) h=mod((c.g-c.b)/d,6.0); else if(cMax==c.g) h=(c.b-c.r)/d+2.0; else h=(c.r-c.g)/d+4.0; h/=6.0; if(h<0.) h+=1.0; }
  float s=(cMax<=0.)?0.:(d/cMax); return vec3(h,s,cMax); }
vec3 hsv2rgb(vec3 c){ float h=c.x*6.0,s=c.y,v=c.z; float i=floor(h),f=h-i;
  float p=v*(1.0-s),q=v*(1.0-s*f),t=v*(1.0-s*(1.0-f));
  if(i==0.0) return vec3(v,t,p); if(i==1.0) return vec3(q,v,p); if(i==2.0) return vec3(p,v,t);
  if(i==3.0) return vec3(p,q,v); if(i==4.0) return vec3(t,p,v); return vec3(v,p,q); }

float hash(vec2 p){ return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453); }
float noise(vec2 p){ vec2 i=floor(p), f=fract(p); f=f*f*(3.0-2.0*f);
  return mix(mix(hash(i),hash(i+vec2(1,0)),f.x), mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x), f.y); }
float fbm(vec2 p){ float v=0.,a=.5; for(int k=0;k<5;k++){ v+=a*noise(p); p=p*2.03+11.7; a*=.5; } return v; }

vec3 sceneNebula(vec2 uv, float t){
  vec2 p = uv*vec2(uCanvasSize.x/uCanvasSize.y,1.0)*2.2;
  float q = fbm(p + vec2(t*0.06, -t*0.03));
  float r = fbm(p*1.6 + q*2.2 + vec2(-t*0.05, t*0.045));
  float s = fbm(p*0.8 - r*1.8 + vec2(t*0.02, 0.));
  vec3 deep = vec3(0.02,0.03,0.08), teal = vec3(0.05,0.55,0.62);
  vec3 ember = vec3(0.95,0.42,0.10), white = vec3(1.0,0.92,0.75);
  vec3 col = mix(deep, teal, smoothstep(0.25,0.75,r));
  col = mix(col, ember, pow(smoothstep(0.45,0.95,s*q*2.0),1.5));
  col += white * pow(max(0.,r*s*2.1-0.55),3.0);
  float vig = smoothstep(1.35,0.35,length(uv-0.5)*1.6);
  return col*vig*1.25;
}
vec3 sceneWarp(vec2 uv, float t){
  vec2 p = (uv-0.5)*vec2(uCanvasSize.x/uCanvasSize.y,1.0);
  float a = atan(p.y,p.x), r = length(p);
  vec3 col = vec3(0.01,0.01,0.03);
  for(int i=0;i<3;i++){
    float fi = float(i);
    float speed = 0.55+fi*0.35;
    float rr = fract(r*1.2 - t*speed*0.25 + fi*0.33);
    float star = noise(vec2(a*22.0+fi*57.0, floor(r*1.2 - t*speed*0.25 + fi*0.33)*7.0));
    star = step(0.88-fi*0.03, star);
    float streak = star * smoothstep(0.0,0.15,rr)*smoothstep(1.0,0.4,rr);
    vec3 tint = mix(vec3(0.55,0.75,1.0), vec3(1.0,0.55,0.85), fi*0.5);
    col += tint * streak * (0.4+r*1.6);
  }
  col += vec3(0.5,0.65,1.0)*exp(-r*6.0)*0.6;
  return col;
}
vec3 sceneTunnel(vec2 uv, float t){
  vec2 p = (uv-0.5)*vec2(uCanvasSize.x/uCanvasSize.y,1.0);
  p += 0.08*vec2(sin(t*0.6), cos(t*0.45));
  float a = atan(p.y,p.x)/6.2831+0.5, r = length(p);
  float z = 0.25/max(0.01,r) + t*1.1;
  float ring = smoothstep(0.42,0.5,abs(fract(z)-0.5));
  float seg = smoothstep(0.46,0.5,abs(fract(a*10.0 + sin(floor(z))*0.2)-0.5));
  float glow2 = exp(-fract(z)*2.0);
  vec3 base = mix(vec3(0.85,0.25,0.05), vec3(1.0,0.75,0.15), fract(z*0.13+a));
  vec3 col = base*(ring*0.9+seg*0.35+glow2*0.35);
  col *= smoothstep(0.0,0.12,r);
  col += vec3(1.0,0.5,0.1)*exp(-r*9.0);
  return col*1.15;
}
vec3 sceneRidges(vec2 uv, float t){
  // vol en parallaxe au-dessus de cretes successives, soleil couchant
  vec3 skyTop=vec3(0.03,0.05,0.13), amber=vec3(0.95,0.52,0.18);
  vec3 col = mix(amber, skyTop, pow(clamp(uv.y*1.25,0.,1.),0.75));
  float ar=uCanvasSize.x/uCanvasSize.y;
  vec2 sunP=vec2(0.62,0.50);
  float sd=length((uv-sunP)*vec2(ar,1.));
  col += vec3(1.0,0.72,0.32)*exp(-sd*sd*26.0)*1.25;
  col += vec3(0.9,0.45,0.15)*exp(-sd*2.6)*0.35;
  for(int i=0;i<5;i++){
    float fi=float(i);
    float speed=0.015+fi*fi*0.012;
    float x=uv.x*(0.7+fi*0.4)+t*speed+fi*4.7;
    float hgt=0.52-fi*0.085+0.11*fbm(vec2(x*2.6, fi*7.7));
    float m=smoothstep(hgt+0.003,hgt-0.003,uv.y);
    vec3 rc=mix(vec3(0.72,0.34,0.17), vec3(0.02,0.02,0.05), fi/4.0);
    col=mix(col, rc, m*(0.55+0.45*fi/4.0));
  }
  return col;
}
vec3 sceneAurora(vec2 uv, float t){
  vec3 col=vec3(0.008,0.015,0.045);
  float st=step(0.996,hash(floor(uv*uCanvasSize/9.0)));
  col+=vec3(st)*0.55*(0.5+0.5*sin(t*2.5+uv.x*60.0));
  float ar=uCanvasSize.x/uCanvasSize.y;
  for(int i=0;i<3;i++){
    float fi=float(i);
    float x=uv.x*ar*1.8+fi*1.7;
    float w=fbm(vec2(x*1.2-t*0.14, fi*3.1+t*0.045));
    float center=0.36+0.36*w;
    float d=uv.y-center;
    float band=exp(-d*d*90.0)*smoothstep(0.0,0.3,uv.y);
    float tail=exp(-max(0.,d)*7.0)*0.35*smoothstep(0.0,0.3,uv.y);
    float rays=0.45+0.55*pow(fbm(vec2(x*7.0, t*0.25+fi)),1.5);
    float flick=0.55+0.45*fbm(vec2(x*3.2+t*0.4, uv.y*2.0-t*0.1));
    vec3 tint=mix(vec3(0.10,0.95,0.45), vec3(0.55,0.20,0.95), fi*0.5);
    col+=tint*(band+tail)*rays*flick*0.9;
  }
  float hgt=0.15+0.08*fbm(vec2(uv.x*4.0,2.3));
  col*=smoothstep(hgt-0.004,hgt+0.010,uv.y);
  return col;
}
vec3 sceneRain(vec2 uv, float t){
  // pluie de glyphes alignee sur la grille ASCII
  float colId=floor(uv.x*uCols);
  float speed=0.12+0.45*hash(vec2(colId,1.0));
  float offset=hash(vec2(colId,7.0))*10.0;
  float y=fract(uv.y+t*speed+offset);
  float head=smoothstep(0.10,0.0,y);
  float trail=smoothstep(0.7,0.0,y)*0.55;
  float on=step(0.12,hash(vec2(colId,floor((t*speed+offset)*2.0))));
  vec3 col=(vec3(0.10,1.0,0.42)*trail+vec3(0.75,1.0,0.85)*head)*on;
  float depth=0.5+0.5*hash(vec2(colId,3.0));
  return col*depth;
}
vec3 sceneLava(vec2 uv, float t){
  vec2 p=(uv-0.5)*vec2(uCanvasSize.x/uCanvasSize.y,1.0);
  float f=0.0;
  for(int i=0;i<6;i++){
    float fi=float(i);
    vec2 c=0.36*vec2(sin(t*(0.30+fi*0.07)+fi*2.1), cos(t*(0.22+fi*0.05)+fi*1.3));
    f+=0.020/max(0.001,dot(p-c,p-c));
  }
  vec3 col=mix(vec3(0.03,0.01,0.07), vec3(1.0,0.34,0.10), smoothstep(2.0,3.6,f));
  col+=vec3(1.0,0.82,0.30)*smoothstep(5.0,9.0,f);
  col+=vec3(0.38,0.06,0.55)*smoothstep(1.1,2.0,f)*(1.0-smoothstep(2.0,3.6,f));
  return col;
}
// fibres optiques neuronales (portage de la scene "neurones" du renderer brut)
float layerN2(float L){ return L<0.5?5.0:(L>2.5?4.0:7.0); }
float threadY2(float k, float x, float t){
  float seg=clamp((x-0.10)/(0.80/3.0),0.0,2.999);
  float L=floor(seg);
  float u=seg-L;
  float nA=layerN2(L), nB=layerN2(L+1.0);
  float iA=floor(hash(vec2(k*1.7,L*9.3))*nA);
  float iB=floor(hash(vec2(k*1.7,(L+1.0)*9.3))*nB);
  float yA=mix(0.24,0.76,(iA+0.5)/nA);
  float yB=mix(0.24,0.76,(iB+0.5)/nB);
  float e=u*u*u*(u*(u*6.0-15.0)+10.0);
  float y=mix(yA,yB,e);
  y+=0.020*(noise(vec2(x*4.0+k*13.1, t*0.10+k*0.7))-0.5)*2.0*sin(3.1416*u);
  return y;
}
vec3 sceneSynapses(vec2 uv, float t){
  float ar=uCanvasSize.x/uCanvasSize.y;
  vec3 col=vec3(0.0);
  vec3 ivory=vec3(0.92,0.89,0.83);
  vec3 gold=vec3(1.0,0.72,0.30);
  float px1=2.0/uCanvasSize.y;
  for(int ki=0;ki<36;ki++){
    float k=float(ki);
    float z=hash(vec2(k,99.1));
    float y=threadY2(k,uv.x,t);
    float dy=uv.y-y;
    float sig=px1*(1.2+4.5*z*z);
    float line=exp(-dy*dy/(2.0*sig*sig));
    vec3 tint=mix(ivory*vec3(0.72,0.80,1.0),ivory,z);
    col+=tint*line*(0.05+0.16*z);
    for(int b=0;b<2;b++){
      float ph=hash(vec2(k,float(b)*3.7+1.0));
      float sp=0.035+0.045*hash(vec2(k,float(b)+7.7));
      float s=fract(t*sp+ph);
      float e=s*s*(3.0-2.0*s);
      float xb=mix(0.04,0.96,e);
      float yb=threadY2(k,xb,t);
      vec2 d2=vec2((uv.x-xb)*ar,(uv.y-yb));
      float bd=dot(d2,d2);
      float bsig=px1*(2.2+4.0*z);
      float amp=0.35+0.65*z;
      col+=gold*exp(-bd/(2.0*bsig*bsig))*amp*1.6;
      float behind=exp(-max(0.0,xb-uv.x)*22.0)*step(uv.x,xb);
      col+=gold*line*behind*amp*0.55;
    }
  }
  for(int L=0;L<4;L++){
    float fL=float(L);
    float n=layerN2(fL);
    float gx=0.10+0.80*fL/3.0;
    for(int i=0;i<7;i++){
      if(float(i)>=n) continue;
      float yn=mix(0.24,0.76,(float(i)+0.5)/n);
      vec2 d2=vec2((uv.x-gx)*ar,(uv.y-yn));
      float dd=dot(d2,d2);
      col+=vec3(1.0,0.96,0.88)*exp(-dd*5000.0)*0.28;
    }
  }
  col*=smoothstep(1.9,0.6,length(uv-0.5)*1.6);
  return col;
}
vec3 magmaPal(float x){
  // noir violace -> violet -> rouge orange -> or -> blanc chaud (DA Magma)
  vec3 c1=vec3(0.03,0.01,0.07), c2=vec3(0.38,0.06,0.55), c3=vec3(1.0,0.34,0.10), c4=vec3(1.0,0.82,0.30), c5=vec3(1.0,0.97,0.85);
  x=clamp(x,0.,1.);
  if(x<0.25) return mix(c1,c2,x/0.25);
  if(x<0.55) return mix(c2,c3,(x-0.25)/0.30);
  if(x<0.85) return mix(c3,c4,(x-0.55)/0.30);
  return mix(c4,c5,(x-0.85)/0.15);
}
vec3 sceneMandel(vec2 uv, float t){
  // respiration infinie dans la vallee des hippocampes, palette magma
  float ar=uCanvasSize.x/uCanvasSize.y;
  float zt=0.5-0.5*cos(t*0.06);
  float scale=exp(mix(0.2,-6.2,zt));
  float rot=t*0.02;
  vec2 p=(uv-0.5)*vec2(ar,1.0)*3.0*scale;
  p=mat2(cos(rot),-sin(rot),sin(rot),cos(rot))*p;
  vec2 c=vec2(-0.743643,0.131825)+p;
  vec2 z=vec2(0.0);
  float n=0.0, m2=0.0;
  for(int i=0;i<110;i++){
    z=vec2(z.x*z.x-z.y*z.y, 2.0*z.x*z.y)+c;
    m2=dot(z,z);
    if(m2>64.0) break;
    n+=1.0;
  }
  if(m2<=64.0) return vec3(0.02,0.0,0.05);
  float sn=n-log2(log2(m2))+4.0;
  float x=0.5+0.5*sin(sn*0.22-t*0.35);
  vec3 col=magmaPal(pow(x,0.9));
  // lueur d'ensemble : les zones a forte iteration irradient
  col+=vec3(1.0,0.5,0.15)*smoothstep(60.0,105.0,n)*0.5;
  return col;
}
vec3 sceneJulia(vec2 uv, float t){
  // julia morphing : c orbite autour de la cardioide, la fractale se metamorphose
  // pendant que la camera derive a l'interieur (zoom respirant + pan lissajous + rotation)
  float ar=uCanvasSize.x/uCanvasSize.y;
  float zoom=exp(mix(0.30,-1.0, 0.5-0.5*cos(t*0.05)));
  vec2 pan=vec2(0.30*sin(t*0.031), 0.24*sin(t*0.043+1.7))*(1.0-zoom*0.6);
  float rot=t*0.015+0.2*sin(t*0.05);
  vec2 z=(uv-0.5)*vec2(ar,1.0)*3.2*zoom;
  z=mat2(cos(rot),-sin(rot),sin(rot),cos(rot))*z+pan;
  vec2 c=uJuliaC; // pilote par la souris (lisse en JS), orbite autonome sinon
  float n=0.0, m2=0.0, trap=1e9;
  for(int i=0;i<110;i++){
    z=vec2(z.x*z.x-z.y*z.y, 2.0*z.x*z.y)+c;
    m2=dot(z,z);
    trap=min(trap,m2);
    if(m2>64.0) break;
    n+=1.0;
  }
  if(m2<=64.0){
    // interieur : orbit trap -> braises sombres qui revelent la structure interne
    float v=pow(clamp(1.0-sqrt(sqrt(trap)),0.,1.),2.2);
    return magmaPal(0.08+v*0.7)*(0.35+0.85*v);
  }
  float sn=n-log2(log2(m2))+4.0;
  float x=0.5+0.5*sin(sn*0.25-t*0.3);
  vec3 col=magmaPal(pow(x,0.9));
  // le champ lointain (faible iteration) reste sombre : seules les structures brulent
  col*=0.12+0.88*smoothstep(2.0,17.0,sn);
  col+=vec3(1.0,0.5,0.15)*smoothstep(45.0,100.0,n)*0.45;
  return col;
}
vec3 sceneImage(vec2 uv, float t){
  // cover-crop + lente respiration (ken burns), brume qui derive sur les cretes
  vec2 c = (uCropMin+uCropMax)*0.5;
  float z = 1.0 + 0.05 + 0.04*sin(t*0.07);
  vec2 uvd = mix(uCropMin, uCropMax, uv);
  uvd = c + (uvd - c)/z;
  vec3 col = texture2D(uImage, vec2(uvd.x, 1.0-uvd.y)).rgb;
  float mist = fbm(uv*vec2(3.0,5.0)+vec2(t*0.06, t*0.01));
  col = mix(col, vec3(0.88,0.93,1.0), 0.16*smoothstep(0.55,0.95,mist));
  col *= 0.90+0.18*fbm(uv*2.2 - vec2(t*0.03,0.));
  return col;
}
vec3 sceneColor(vec2 uv){
  if(uUseImage==1) return sceneImage(uv,uTime);
  if(uScene==1) return sceneWarp(uv,uTime);
  if(uScene==2) return sceneTunnel(uv,uTime);
  if(uScene==3) return sceneRidges(uv,uTime);
  if(uScene==4) return sceneAurora(uv,uTime);
  if(uScene==5) return sceneRain(uv,uTime);
  if(uScene==6) return sceneLava(uv,uTime);
  if(uScene==7) return sceneMandel(uv,uTime);
  if(uScene==8) return sceneJulia(uv,uTime);
  if(uScene==9) return sceneSynapses(uv,uTime);
  return sceneNebula(uv,uTime);
}

float glowAmount(vec2 tileCenterUV){
  float combined = 0.0;
  for(int i=0;i<MAX_GLOW;i++){
    if(i<uGlowCount){
      float age = uNow - uGlowStart[i];
      float timeF = clamp(1.0 - age/max(0.0001,uGlowDuration), 0.0, 1.0);
      vec2 dpPx = (tileCenterUV - uGlowCenters[i]) * uCanvasSize;
      float distPx = length(dpPx);
      float r = max(1.0, uGlowRadiiPx[i]);
      float sepPx = length((uGlowCenters[i]-uGlowCenters[0])*uCanvasSize);
      float shrinkK = max(0.0,uGlowShrinkStrength);
      float shrinkFactor = (shrinkK>0.0)? max(0.15, exp(-shrinkK*sepPx/r)) : 1.0;
      float rScaled = r*shrinkFactor;
      float r0 = clamp(uGlowInnerFrac,0.0,0.99)*rScaled;
      float spatial;
      if(distPx<=r0){ spatial=1.0; }
      else{ float tt=clamp((distPx-r0)/max(0.0001,(rScaled-r0)),0.0,1.0);
        spatial=pow(1.0-tt, max(0.0001,uGlowFalloffExp)); }
      combined = min(1.0, combined + spatial*timeF);
    }
  }
  return combined;
}
vec3 applyGlowColor(vec3 baseColor, float combined){
  if(combined<=0.0) return baseColor;
  vec3 top = (uGlowUseTileColor==1)? clamp(baseColor*uGlowIntensity,0.,1.) : vec3(clamp(uGlowIntensity*combined,0.,1.));
  vec3 glowed = doBlend(baseColor, top, uGlowBlendMode);
  vec3 mixed = mix(baseColor, glowed, clamp(uGlowOpacity*combined,0.,1.));
  if(uGlowSaturationBoost>0.001){
    vec3 hsv = rgb2hsv(mixed);
    hsv.y = clamp(hsv.y*(1.0+uGlowSaturationBoost*combined),0.,1.);
    mixed = hsv2rgb(hsv);
  }
  return mixed;
}

void main(){
  vec2 grid = vec2(uCols,uRows);
  vec2 cellCoord = floor(vUV*grid);
  vec2 cellUV = fract(vUV*grid);
  vec2 tileCenterUV = (cellCoord+0.5)*uInvGrid;

  vec3 videoRGB = sceneColor(tileCenterUV);
  float lum = pow(clamp(luminance(videoRGB),0.,1.), uGamma);

  float gAmt = 0.0;
  if(uGlowCount>0){ gAmt = glowAmount(tileCenterUV); }
  float lumAdj = lum;
  if(gAmt>0.0){
    float shaped = pow(gAmt, max(0.0001,uGlowLumaExp));
    lumAdj = clamp(lum + shaped*uGlowLumaGain, 0., 1.);
  }
  float glyphIdx = floor(lumAdj*(uGlyphCount-1.0)+0.5);

  float charH = uCellSizePx.y*uCharFillRatio;
  float charW = charH*uCharAspectRatio;
  vec2 charSizeInCell = vec2(charW/uCellSizePx.x, charH/uCellSizePx.y);
  vec2 charStart = (1.0-charSizeInCell)*0.5;
  vec2 charEnd = charStart+charSizeInCell;
  float charMask = 1.0;
  if(cellUV.x<charStart.x||cellUV.x>charEnd.x||cellUV.y<charStart.y||cellUV.y>charEnd.y) charMask=0.0;
  vec2 charUV = clamp((cellUV-charStart)/charSizeInCell, 0., 1.);

  float aCols = uAtlasGrid.x;
  float ax = mod(glyphIdx,aCols);
  float ay = floor(glyphIdx/aCols);
  vec2 atlasUV = (vec2(ax,ay)+charUV)/uAtlasGrid;
  float glyphAlpha = texture2D(uAtlas, atlasUV).r * charMask * uGlyphOpacity;

  vec3 baseRGB = videoRGB * clamp(uTileOpacity,0.,1.);
  vec3 blended = doBlend(baseRGB, videoRGB, uBlendMode);
  if(uBlendMode!=3) blended = mix(baseRGB, blended, clamp(uBlendStrength,0.,1.));
  if(uPreserveHue==1) blended = retintToBase(blended, baseRGB);
  vec3 finalRGB = mix(baseRGB, blended, glyphAlpha);
  if(uGlowCount>0 && gAmt>0.0) finalRGB = applyGlowColor(finalRGB, gAmt);
  gl_FragColor = vec4(finalRGB,1.0);
}`;

export type Atlas = { canvas: HTMLCanvasElement; gridCols: number; gridRows: number };
let atlasPromise: Promise<Atlas> | null = null;

export function buildAtlas(): Promise<Atlas> {
  atlasPromise ??= (async () => {
    try { await document.fonts.load("300 50px 'GeistMono-Light'"); } catch {}
    const n = DENSITY.length;
    const gcols = Math.ceil(Math.sqrt(n)), grows = Math.ceil(n / gcols);
    const cell = CFG.atlasGlyphSize;
    const cv = document.createElement("canvas");
    cv.width = gcols * cell; cv.height = grows * cell;
    const g = cv.getContext("2d")!;
    g.fillStyle = "#000"; g.fillRect(0, 0, cv.width, cv.height);
    g.fillStyle = "#fff"; g.textAlign = "center"; g.textBaseline = "middle";
    g.font = `300 ${Math.floor(cell * 0.82)}px 'GeistMono-Light', monospace`;
    for (let i = 0; i < n; i++) {
      const ch = DENSITY[n - 1 - i];
      g.fillText(ch, (i % gcols) * cell + cell / 2, Math.floor(i / gcols) * cell + cell / 2 + cell * 0.02);
    }
    return { canvas: cv, gridCols: gcols, gridRows: grows };
  })();
  return atlasPromise;
}

const renderers: AsciiRenderer[] = [];
let hubStarted = false;
let loopStarted = false;
let hasPointer = false;
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

class AsciiRenderer {
  wrap: HTMLElement; canvas: HTMLCanvasElement; scene: number;
  gl!: WebGLRenderingContext; prog!: WebGLProgram; u: Record<string, WebGLUniformLocation | null> = {};
  glow: { x: number; y: number; rPx: number; tSec: number }[] = [];
  lastStamp = 0; cellW = 10; dpr: number; ready = false; ambient: boolean;
  imgAspect = 0;
  mouse: { x: number; y: number } | null = null;
  jc = { x: 0.7885, y: 0 };
  private centers = new Float32Array(16);
  private radii = new Float32Array(8);
  private starts = new Float32Array(8);

  constructor(wrap: HTMLElement, scene: number, imageUrl?: string) {
    this.wrap = wrap; this.scene = scene;
    this.ambient = wrap.hasAttribute("data-ascii-ambient");
    this.canvas = wrap.querySelector("canvas")!;
    this.dpr = Math.max(1, Math.min(CFG.dprCap, devicePixelRatio || 1));
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
    for (const m of FS.matchAll(/uniform\s+\w+\s+(\w+)/g)) this.u[m[1]] = gl.getUniformLocation(p, m[1]);

    gl.uniform1f(this.u.uCharAspectRatio, CFG.charAspectRatio);
    gl.uniform1f(this.u.uCharFillRatio, CFG.charFillRatio);
    gl.uniform1f(this.u.uGamma, CFG.gamma);
    gl.uniform1f(this.u.uTileOpacity, CFG.tileOpacity);
    gl.uniform1f(this.u.uGlyphOpacity, CFG.glyphOpacity);
    gl.uniform1i(this.u.uBlendMode, CFG.blendMode);
    gl.uniform1f(this.u.uBlendStrength, CFG.blendStrength);
    gl.uniform1i(this.u.uPreserveHue, CFG.preserveHue);
    gl.uniform1f(this.u.uGlyphCount, DENSITY.length);
    gl.uniform1i(this.u.uScene, scene);
    gl.uniform1f(this.u.uGlowDuration, CFG.glowDurationSec);
    gl.uniform1f(this.u.uGlowIntensity, CFG.glowIntensity);
    gl.uniform1f(this.u.uGlowOpacity, CFG.glowOpacity);
    gl.uniform1i(this.u.uGlowBlendMode, CFG.glowBlendMode);
    gl.uniform1i(this.u.uGlowUseTileColor, CFG.glowUseTileColor);
    gl.uniform1f(this.u.uGlowSaturationBoost, CFG.glowSaturationBoost);
    gl.uniform1f(this.u.uGlowInnerFrac, CFG.glowInnerFrac);
    gl.uniform1f(this.u.uGlowFalloffExp, CFG.glowFalloffExp);
    gl.uniform1f(this.u.uGlowLumaGain, CFG.glowLumaGain);
    gl.uniform1f(this.u.uGlowLumaExp, CFG.glowLumaExp);
    gl.uniform1f(this.u.uGlowShrinkStrength, CFG.glowShrinkStrength);
    gl.uniform1i(this.u.uGlowCount, 0);
    gl.uniform1i(this.u.uUseImage, 0);
    gl.uniform2f(this.u.uCropMin, 0, 0);
    gl.uniform2f(this.u.uCropMax, 1, 1);
    if (imageUrl) this.loadImage(imageUrl);

    new ResizeObserver(() => this.resize()).observe(this.wrap);
    this.resize();
    renderers.push(this);
    buildAtlas().then((a) => { this.setAtlas(a); requestAnimationFrame(() => this.wrap.classList.add("active")); });
  }

  loadImage(url: string) {
    const img = new Image();
    img.onload = () => {
      const gl = this.gl;
      gl.useProgram(this.prog);
      const tex = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
      gl.uniform1i(this.u.uImage, 0);
      gl.uniform1i(this.u.uUseImage, 1);
      this.imgAspect = img.naturalWidth / img.naturalHeight;
      this.updateCrop();
    };
    img.src = url;
  }

  // cover-crop UV (même logique que _computeCoverCropUV du site original)
  updateCrop() {
    if (!this.imgAspect) return;
    const tgtAR = this.canvas.width / Math.max(1, this.canvas.height);
    let uMin = 0, uMax = 1, vMin = 0, vMax = 1;
    if (this.imgAspect > tgtAR) {
      const visW = tgtAR / this.imgAspect;
      uMin = (1 - visW) * 0.5; uMax = uMin + visW;
    } else if (this.imgAspect < tgtAR) {
      const visH = this.imgAspect / tgtAR;
      vMin = (1 - visH) * 0.35; vMax = vMin + visH; // focal un peu vers le haut (sommets)
    }
    const gl = this.gl;
    gl.useProgram(this.prog);
    gl.uniform2f(this.u.uCropMin, uMin, vMin);
    gl.uniform2f(this.u.uCropMax, uMax, vMax);
  }

  setAtlas(atlas: Atlas) {
    const gl = this.gl;
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
    this.ready = true;
  }

  resize() {
    const r = this.wrap.getBoundingClientRect();
    const w = Math.max(2, Math.floor(r.width * this.dpr)), h = Math.max(2, Math.floor(r.height * this.dpr));
    if (this.canvas.width === w && this.canvas.height === h) return;
    this.canvas.width = w; this.canvas.height = h;
    const gl = this.gl;
    gl.viewport(0, 0, w, h); gl.useProgram(this.prog);
    const cellDev = CFG.cellSizePx * this.dpr;
    const cols = Math.max(10, Math.min(256, Math.floor(w / cellDev)));
    const cellW = w / cols, cellH = cellW / CFG.cellAspectRatio;
    const rows = Math.max(5, Math.floor(h / cellH));
    this.cellW = cellW;
    gl.uniform2f(this.u.uCanvasSize, w, h);
    gl.uniform1f(this.u.uCols, cols); gl.uniform1f(this.u.uRows, rows);
    gl.uniform2f(this.u.uInvGrid, 1 / cols, 1 / rows);
    gl.uniform2f(this.u.uCellSizePx, cellW, Math.max(1, h / rows));
    this.updateCrop();
  }

  pointer(clientX: number, clientY: number, nowMs: number) {
    const r = this.wrap.getBoundingClientRect();
    if (this.scene === 8) {
      // le morphing de julia suit la souris partout dans la fenetre, clampe au cadre
      this.mouse = {
        x: Math.max(0, Math.min(1, (clientX - r.left) / r.width)),
        y: Math.max(0, Math.min(1, (clientY - r.top) / r.height)),
      };
    }
    if (clientX < r.left - 80 || clientX > r.right + 80 || clientY < r.top - 80 || clientY > r.bottom + 80) return;
    if (nowMs - this.lastStamp < CFG.glowStampIntervalMs) return;
    this.lastStamp = nowMs;
    this.glow.unshift({
      x: (clientX - r.left) / r.width, y: 1 - (clientY - r.top) / r.height,
      rPx: CFG.glowRadiusMultiplier * this.cellW, tSec: nowMs / 1000,
    });
    if (this.glow.length > 8) this.glow.length = 8;
  }

  draw(tSec: number) {
    if (!this.ready) return;
    const gl = this.gl;
    gl.useProgram(this.prog);
    this.glow = this.glow.filter((p) => tSec - p.tSec <= CFG.glowDurationSec + 0.25);
    const n = Math.min(8, this.glow.length);
    for (let i = 0; i < n; i++) {
      const p = this.glow[i];
      this.centers[i * 2] = p.x; this.centers[i * 2 + 1] = p.y;
      this.radii[i] = p.rPx; this.starts[i] = p.tSec;
    }
    gl.uniform1i(this.u.uGlowCount, n);
    if (n) {
      gl.uniform2fv(this.u.uGlowCenters, this.centers);
      gl.uniform1fv(this.u.uGlowRadiiPx, this.radii);
      gl.uniform1fv(this.u.uGlowStart, this.starts);
    }
    if (this.scene === 8) {
      // cible : souris → plan complexe (zone riche autour de la cardioide), sinon orbite
      let tx: number, ty: number;
      if (this.mouse) {
        // la souris fixe le point de base, une petite orbite continue de morpher autour
        const th = tSec * 0.35;
        tx = -0.85 + this.mouse.x * 1.3 + 0.055 * Math.cos(th);
        ty = -0.75 + this.mouse.y * 1.5 + 0.055 * Math.sin(th * 0.83 + 1.2);
      } else {
        const th = tSec * 0.12;
        tx = 0.7885 * Math.cos(th); ty = 0.7885 * Math.sin(th);
      }
      this.jc.x += (tx - this.jc.x) * 0.045;
      this.jc.y += (ty - this.jc.y) * 0.045;
      gl.uniform2f(this.u.uJuliaC, this.jc.x, this.jc.y);
      (window as any).__juliaC = { cr: this.jc.x, ci: this.jc.y, manual: !!this.mouse };
    }
    gl.uniform1f(this.u.uNow, tSec);
    gl.uniform1f(this.u.uTime, reduced() ? 12.0 : tSec);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }
}

function startHub() {
  if (hubStarted) return;
  hubStarted = true;
  let px = 0, py = 0, pending = false;
  addEventListener("pointermove", (e) => {
    px = e.clientX; py = e.clientY; hasPointer = true;
    if (!pending) {
      pending = true;
      requestAnimationFrame(() => {
        pending = false;
        const now = performance.now();
        for (const r of renderers) r.pointer(px, py, now);
      });
    }
  }, { passive: true });
}

function startLoop() {
  if (loopStarted) return;
  loopStarted = true;
  const loop = (now: number) => {
    const t = now / 1000;
    if (!hasPointer && !reduced()) {
      for (const r of renderers) {
        if (!r.ambient) continue;
        const rect = r.wrap.getBoundingClientRect();
        const ax = rect.left + rect.width * (0.5 + 0.33 * Math.sin(t * 0.5) + 0.12 * Math.sin(t * 1.7));
        const ay = rect.top + rect.height * (0.5 + 0.28 * Math.cos(t * 0.4) + 0.1 * Math.cos(t * 1.3));
        r.pointer(ax, ay, now);
      }
    }
    for (const r of renderers) r.draw(t);
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

export function initAscii(wrap: HTMLElement, scene: number, image?: string) {
  new AsciiRenderer(wrap, scene, image);
  startHub();
  startLoop();
}
