// A fetch for decision.js that keeps a model's files in Cache Storage. The first visit downloads each file
// whole (with progress); later visits read it from disk. decision.js's Range requests on weights.bin are
// served by slicing the cached file. Without Cache Storage (private windows), files stay in memory only.

type Progress = (loaded: number, total: number) => void;

export function cachingFetch(cacheName: string, onDownload?: Progress): typeof fetch {
  const files = new Map<string, Promise<Blob>>();

  async function file(url: string): Promise<Blob> {
    const cache = await caches.open(cacheName).catch(() => undefined);
    const hit = await cache?.match(url);
    if (hit) return hit.blob();
    const res = await fetch(url);
    if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
    const total = Number(res.headers.get("content-length")) || 0;
    const chunks: Uint8Array[] = [];
    let loaded = 0;
    for (const reader = res.body.getReader(); ;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (url.endsWith("weights.bin")) onDownload?.(loaded, total);
    }
    const blob = new Blob(chunks as BlobPart[]);
    await cache?.put(url, new Response(blob)).catch(() => undefined); // quota: keep it in memory only
    return blob;
  }

  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input);
    if (!files.has(url)) files.set(url, file(url));
    const blob = await files.get(url)!;
    const range = new Headers(init?.headers).get("Range")?.match(/bytes=(\d+)-(\d+)/);
    return range ? new Response(blob.slice(+range[1]!, +range[2]! + 1), { status: 206 }) : new Response(blob);
  }) as typeof fetch;
}

/** Whether `url` is already in the cache. */
export async function isCached(cacheName: string, url: string): Promise<boolean> {
  const cache = await caches.open(cacheName).catch(() => undefined);
  return !!(await cache?.match(url));
}

/** Frees the cached model. */
export const clearCache = (cacheName: string) => caches.delete(cacheName).catch(() => false);
