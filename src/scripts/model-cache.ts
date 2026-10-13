// A fetch for decision.js that keeps a model's files in the Origin Private File System (OPFS). The first
// visit streams each file straight to disk (with progress); later visits read it from there. decision.js's
// Range requests on weights.bin are served by slicing the file, which reads only those bytes. Without OPFS
// (or without room for the model), files stay in memory for this visit only.

type Progress = (loaded: number, total: number) => void;

// One OPFS directory per model; files are named after their URL. A file counts as cached only once its
// ".ok" marker exists, so a download interrupted halfway is started again.
const dir = (cacheName: string, create = false) =>
  navigator.storage?.getDirectory?.().then((root) => root.getDirectoryHandle(cacheName, { create })).catch(() => undefined);
const fileName = (url: string) => encodeURIComponent(new URL(url).pathname.split("/").pop() || "index");

async function cachedFile(cacheName: string, url: string): Promise<File | undefined> {
  const d = await dir(cacheName);
  if (!d) return undefined;
  try {
    await d.getFileHandle(fileName(url) + ".ok");
    return await (await d.getFileHandle(fileName(url))).getFile();
  } catch {
    return undefined;
  }
}

export function cachingFetch(cacheName: string, onDownload?: Progress): typeof fetch {
  const files = new Map<string, Promise<Blob>>();
  navigator.storage?.persist?.().catch(() => false); // ask the browser not to evict the model under storage pressure
  caches?.delete(cacheName).catch(() => false); // the model used to live in Cache Storage: free that copy

  async function download(url: string, toDisk = true): Promise<Blob> {
    const res = await fetch(url);
    if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
    const total = Number(res.headers.get("content-length")) || 0;
    let loaded = 0;
    const counted = res.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({
      transform(chunk, out) {
        loaded += chunk.length;
        if (url.endsWith("weights.bin")) onDownload?.(loaded, total);
        out.enqueue(chunk);
      },
    }));
    const d = toDisk ? await dir(cacheName, true) : undefined;
    if (!d) return new Response(counted).blob(); // no OPFS: keep it in memory for this visit

    try {
      const handle = await d.getFileHandle(fileName(url), { create: true });
      await counted.pipeTo(await handle.createWritable());
      await d.getFileHandle(fileName(url) + ".ok", { create: true });
      return await handle.getFile();
    } catch {
      await d.removeEntry(fileName(url)).catch(() => undefined); // e.g. no room left: drop the partial file
      return download(url, false);
    }
  }

  return (async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = input instanceof Request ? input.url : String(input);
    if (!files.has(url)) files.set(url, cachedFile(cacheName, url).then((f) => f ?? download(url)));
    const blob = await files.get(url)!;
    const range = new Headers(init?.headers).get("Range")?.match(/bytes=(\d+)-(\d+)/);
    return range ? new Response(blob.slice(+range[1]!, +range[2]! + 1), { status: 206 }) : new Response(blob);
  }) as typeof fetch;
}

/** Whether `url` is already on disk. */
export async function isCached(cacheName: string, url: string): Promise<boolean> {
  return !!(await cachedFile(cacheName, url));
}

/** Frees the stored model. */
export async function clearCache(cacheName: string): Promise<boolean> {
  const root = await navigator.storage?.getDirectory?.().catch(() => undefined);
  return !!(await root?.removeEntry(cacheName, { recursive: true }).then(() => true, () => false));
}
