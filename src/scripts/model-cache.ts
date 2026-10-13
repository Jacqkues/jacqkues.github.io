// A fetch for decision.js that keeps a model's files in the Origin Private File System (OPFS). The first
// visit downloads each file in 64 MB parts, each one saved to disk as soon as it is complete: a dropped
// connection (common on phones) only retries the current part, and a reload resumes where it stopped.
// Later visits read the parts from disk; decision.js's Range requests on weights.bin slice them, which
// reads only those bytes. Without OPFS, files stay in memory for this visit only.

type Progress = (loaded: number, total: number) => void;

const PART = 64 << 20;

// One OPFS directory per model. A file "x" is stored as "x.size" (its length, known after the first
// response) and parts "x.0", "x.1", …; a part counts once it has its full length.
/** Whether this browser can keep the model on disk (some Safari versions cannot write OPFS files). */
export const canStore = typeof FileSystemFileHandle !== "undefined" && "createWritable" in FileSystemFileHandle.prototype;
const dir = (cacheName: string, create = false) =>
  !canStore ? Promise.resolve(undefined) : navigator.storage?.getDirectory?.().then((root) => root.getDirectoryHandle(cacheName, { create })).catch(() => undefined);
const fileName = (url: string) => encodeURIComponent(new URL(url).pathname.split("/").pop() || "index");
const partLength = (size: number, i: number) => Math.min(PART, size - i * PART);

async function readText(d: FileSystemDirectoryHandle, name: string): Promise<string | undefined> {
  return d.getFileHandle(name).then((h) => h.getFile()).then((f) => f.text(), () => undefined);
}

/** The parts already on disk for `url`: the file's size (if known) and how many leading parts are complete. */
async function stored(d: FileSystemDirectoryHandle, url: string): Promise<{ size?: number; parts: File[] }> {
  const name = fileName(url), size = Number(await readText(d, name + ".size")) || undefined, parts: File[] = [];
  if (size === undefined) return { parts };
  for (let i = 0; i * PART < size; i++) {
    const f = await d.getFileHandle(`${name}.${i}`).then((h) => h.getFile(), () => undefined);
    if (!f || f.size !== partLength(size, i)) break;
    parts.push(f);
  }
  return { size, parts };
}

async function cachedFile(cacheName: string, url: string): Promise<Blob | undefined> {
  const d = await dir(cacheName);
  if (!d) return undefined;
  const { size, parts } = await stored(d, url);
  return size !== undefined && parts.length * PART >= size ? new Blob(parts) : undefined;
}

// Waits until the page can download again: online, and visible (phones suspend background tabs).
const settled = () => new Promise<void>((resolve) => {
  const check = () => {
    if (!navigator.onLine || document.visibilityState !== "visible") return;
    removeEventListener("online", check);
    document.removeEventListener("visibilitychange", check);
    resolve();
  };
  addEventListener("online", check);
  document.addEventListener("visibilitychange", check);
  check();
});

// An error of the connection (dropped, stalled), as opposed to one of the server or the disk.
class NetworkError extends Error {}
const STALL = 20_000; // no byte for this long: the connection is stuck (phones), start the request again

/** Fetches `url` and calls `chunk` with each piece of the body; a dropped or stalled connection throws a NetworkError. */
async function stream(url: string, headers: HeadersInit, chunk: (value: Uint8Array<ArrayBuffer>, res: Response) => Promise<void> | void): Promise<Response> {
  const abort = new AbortController();
  let timer = setTimeout(() => abort.abort(), STALL);
  try {
    const res = await fetch(url, { headers, signal: abort.signal }).catch((e) => { throw new NetworkError(String(e)); });
    if (!res.ok || !res.body) throw new Error(`${url}: HTTP ${res.status}`);
    for (const reader = res.body.getReader(); ;) {
      clearTimeout(timer);
      timer = setTimeout(() => abort.abort(), STALL);
      const { done, value } = await reader.read().catch((e) => { throw new NetworkError(String(e)); });
      if (done) return res;
      await chunk(value, res);
    }
  } finally {
    clearTimeout(timer);
  }
}

/** `onDownload` reports the bytes of weights.bin; `onRetry` says when the connection dropped and the download waits to go on. */
export function cachingFetch(cacheName: string, onDownload?: Progress, onRetry?: (attempt: number) => void): typeof fetch {
  const files = new Map<string, Promise<Blob>>();
  navigator.storage?.persist?.().catch(() => false); // ask the browser not to evict the model under storage pressure
  caches?.delete(cacheName).catch(() => false); // the model used to live in Cache Storage: free that copy
  const progress = (url: string, loaded: number, total: number) => url.endsWith("weights.bin") && onDownload?.(loaded, total);

  async function inMemory(url: string): Promise<Blob> {
    const chunks: Uint8Array<ArrayBuffer>[] = [];
    let loaded = 0;
    await stream(url, {}, (v, res) => { chunks.push(v); progress(url, (loaded += v.length), Number(res.headers.get("content-length"))); });
    return new Blob(chunks as BlobPart[]);
  }

  // Downloads part i (bytes [i·PART, …)) into its own file; returns the file's total size.
  async function fetchPart(d: FileSystemDirectoryHandle, url: string, i: number, done: number, size?: number): Promise<number> {
    const end = size === undefined ? (i + 1) * PART : i * PART + partLength(size, i);
    const w = await (await d.getFileHandle(`${fileName(url)}.${i}`, { create: true })).createWritable();
    // 206 says the file's size; 200 (the whole file) is fine only for a small file, in one part
    const sizeOf = (res: Response) => res.status === 206 ? Number(res.headers.get("content-range")?.split("/")[1]) : i === 0 ? Number(res.headers.get("content-length")) : 0;
    let loaded = 0, total = 0;
    try {
      const res = await stream(url, { Range: `bytes=${i * PART}-${end - 1}` }, async (v, res) => {
        total ||= sizeOf(res);
        if (!total || (res.status === 200 && total > PART)) throw new Error(`${url}: the server does not support Range requests`);
        await w.write(v);
        progress(url, done + (loaded += v.length), total);
      });
      total ||= sizeOf(res) || loaded; // an empty file never calls the callback
      await w.close(); // the part is saved only here
    } catch (e) {
      await w.abort().catch(() => undefined);
      throw e;
    }
    if (size === undefined) {
      const s = await (await d.getFileHandle(fileName(url) + ".size", { create: true })).createWritable();
      await s.write(String(total));
      await s.close();
    }
    return total;
  }

  async function toDisk(d: FileSystemDirectoryHandle, url: string): Promise<Blob> {
    for (const old of [fileName(url), fileName(url) + ".ok"]) await d.removeEntry(old).catch(() => undefined); // the earlier one-file layout
    let { size, parts } = await stored(d, url);
    for (let failures = 0; size === undefined || parts.length * PART < size;) {
      try {
        size = await fetchPart(d, url, parts.length, parts.length * PART, size);
        ({ parts } = await stored(d, url));
        failures = 0;
      } catch (e) {
        if (!(e instanceof NetworkError) || ++failures > 8) throw e;
        onRetry?.(failures);
        await settled(); // then retry the same part, after a short backoff
        await new Promise((r) => setTimeout(r, 500 * 2 ** Math.min(failures, 5)));
      }
    }
    return new Blob(parts);
  }

  async function download(url: string): Promise<Blob> {
    const d = await dir(cacheName, true);
    const wake = await navigator.wakeLock?.request("screen").catch(() => undefined); // a sleeping phone drops the connection
    try {
      if (!d) return await inMemory(url);
      return await toDisk(d, url);
    } catch (e) {
      // the disk failed (full, or a browser that cannot write files): keep the model in memory for this visit
      if (d && !(e instanceof NetworkError) && !/HTTP|Range/.test(String(e))) return await inMemory(url);
      throw e;
    } finally {
      wake?.release().catch(() => undefined);
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

/** Whether `url` is fully on disk. */
export async function isCached(cacheName: string, url: string): Promise<boolean> {
  return !!(await cachedFile(cacheName, url));
}

/** How much of `url` is already on disk, for a download stopped halfway (0 when nothing or all of it is). */
export async function partial(cacheName: string, url: string): Promise<{ loaded: number; total: number }> {
  const d = await dir(cacheName);
  const { size, parts } = d ? await stored(d, url) : { parts: [] };
  const loaded = parts.reduce((a, f) => a + f.size, 0);
  return size && loaded < size ? { loaded, total: size } : { loaded: 0, total: 0 };
}

/** Frees the stored model. */
export async function clearCache(cacheName: string): Promise<boolean> {
  const root = await navigator.storage?.getDirectory?.().catch(() => undefined);
  return !!(await root?.removeEntry(cacheName, { recursive: true }).then(() => true, () => false));
}
