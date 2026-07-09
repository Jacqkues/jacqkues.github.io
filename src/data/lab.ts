export type LabVariant = {
  slug: string;
  name: string;
  tagline: string;
  scene: number;
  image?: string;
  hud?: "mandel" | "julia";
  raw?: number; // scene du renderer brut (sans ASCII)
  trailRad?: number;
};

export const VARIANTS: LabVariant[] = [
  { slug: "sommets", name: "Sommets", tagline: "Crêtes génératives, brume et couchant.", scene: 0, image: "/ridges.jpg" },
  { slug: "vignemale", name: "Vignemale", tagline: "3298 m, face nord en glyphes.", scene: 0, image: "/vignemale.jpg" },
  { slug: "traversee", name: "Traversée", tagline: "Vol parallaxe au-dessus des crêtes, en pur shader.", scene: 3 },
  { slug: "aurore", name: "Aurore", tagline: "Rideaux magnétiques sur ciel étoilé.", scene: 4 },
  { slug: "signal", name: "Signal", tagline: "Pluie de code alignée sur la grille.", scene: 5 },
  { slug: "fractale", name: "Fractale", tagline: "Respiration infinie dans l'ensemble de Mandelbrot, palette magma.", scene: 7, hud: "mandel" },
  { slug: "julia", name: "Julia", tagline: "c orbite autour de la cardioïde — la fractale se métamorphose.", scene: 8, hud: "julia" },
  { slug: "magma", name: "Magma", tagline: "Metaballs incandescentes, lave organique.", scene: 6 },
  { slug: "nebuleuse", name: "Nébuleuse", tagline: "Champs de bruit fractal, braise et sarcelle.", scene: 0 },
  { slug: "warp", name: "Warp", tagline: "Champ d'étoiles en hyperespace.", scene: 1 },
  { slug: "tunnel", name: "Tunnel", tagline: "Anneaux néon en fuite perpétuelle.", scene: 2 },
  { slug: "liquide", name: "Liquide", tagline: "Métal en fusion — metaballs 3D raymarchées, la souris déplace la lumière.", scene: 0, raw: 0 },
  { slug: "menger", name: "Menger", tagline: "Éponge fractale 3D, raymarching et brouillard de braise.", scene: 0, raw: 1 },
  { slug: "flux", name: "Flux", tagline: "Domain warping pleine résolution — la souris tord le champ.", scene: 0, raw: 2 },
  { slug: "caustiques", name: "Caustiques", tagline: "Eau incandescente, interférences lumineuses.", scene: 0, raw: 3 },
  { slug: "neurones", name: "Neurones", tagline: "Faisceaux de fibres optiques — les signaux voyagent de neurone en neurone.", scene: 0, raw: 9 },
  { slug: "cerveau", name: "Cerveau", tagline: "Silhouette abstraite, circonvolutions irisées — une pensée traverse le cortex.", scene: 0, raw: 11 },
  { slug: "cortex", name: "Cortex", tagline: "Le réseau en volume — caméra orbitale, bokeh, brouillard de profondeur.", scene: 0, raw: 10 },
  { slug: "synapses", name: "Synapses", tagline: "Les fibres neuronales, traduites en glyphes — le mélange des deux renderers.", scene: 9 },
  { slug: "descente", name: "Descente", tagline: "Loss surface en topographie — la bille suit le gradient, converge, repart.", scene: 0, raw: 8, trailRad: 0.028 },
  { slug: "encre", name: "Encre", tagline: "Marbrure sumi-e — voiles d'encre blanche sur papier noir.", scene: 0, raw: 4 },
  { slug: "topographie", name: "Topographie", tagline: "Courbes de niveau d'un terrain qui n'existe pas.", scene: 0, raw: 5 },
  { slug: "soie", name: "Soie", tagline: "Filaments irisés, reflets moirés — le champ tissé.", scene: 0, raw: 6 },
  { slug: "abysse", name: "Abysse", tagline: "Bioluminescence des profondeurs, plancton scintillant.", scene: 0, raw: 7 },
];
