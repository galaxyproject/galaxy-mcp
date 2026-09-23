import { describe, it, expect } from "vitest";
import { createGalaxyContext, DEFAULT_POLL } from "../src/context";

describe("createGalaxyContext", () => {
  it("builds a client and applies poll defaults", () => {
    const ctx = createGalaxyContext({ baseUrl: "https://g.example", apiKey: "K" });
    expect(ctx.client).toBeDefined();
    expect(ctx.poll).toEqual(DEFAULT_POLL);
  });
  it("allows poll overrides and passes through a signal", () => {
    const ac = new AbortController();
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      poll: { intervalMs: 50, timeoutMs: 1000 },
      signal: ac.signal,
    });
    expect(ctx.poll.intervalMs).toBe(50);
    expect(ctx.poll.timeoutMs).toBe(1000);
    expect(ctx.poll.backoff).toBe(DEFAULT_POLL.backoff); // unspecified -> default
    expect(ctx.signal).toBe(ac.signal);
  });
  it("stores baseUrl on the context for surfaces that need it", () => {
    const ctx = createGalaxyContext({ baseUrl: "https://g.example", apiKey: "K" });
    expect(ctx.baseUrl).toBe("https://g.example");
  });
});

/** Never answers on its own; settles only when whoever called it gives up. */
function hangingFetch() {
  return (async (req: Request, init?: RequestInit) => {
    const signal = init?.signal ?? req.signal;
    await new Promise((_resolve, reject) => {
      if (signal?.aborted) return reject(signal.reason);
      signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
    });
    throw new Error("unreachable");
  }) as unknown as typeof fetch;
}

describe("the context's Galaxy version lookup", () => {

  const versionFetch = (body: unknown, status = 200) => {
    const calls: string[] = [];
    const fetchImpl = (async (req: Request | string | URL) => {
      calls.push(req instanceof Request ? req.url : String(req));
      return new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" },
      });
    }) as unknown as typeof fetch;
    return { calls, fetchImpl };
  };

  it("costs nothing until something asks", async () => {
    const { calls, fetchImpl } = versionFetch({ version_major: "26.1" });
    createGalaxyContext({ baseUrl: "https://g.example", apiKey: "K", fetchImpl });
    await new Promise((resolve) => setTimeout(resolve, 0)); // past the microtask queue too
    expect(calls).toEqual([]);
  });

  it("asks once and hands the same answer to everyone after", async () => {
    const { calls, fetchImpl } = versionFetch({ version_major: "26.1", version_minor: "1" });
    const ctx = createGalaxyContext({ baseUrl: "https://g.example", apiKey: "K", fetchImpl });
    const [first, second] = await Promise.all([ctx.galaxyVersion!(), ctx.galaxyVersion!()]);
    expect(first.version).toMatchObject({ major: 26, minor: 1 });
    expect(first.payload).toEqual({ version_major: "26.1", version_minor: "1" });
    expect(second).toEqual(first);
    expect(await ctx.galaxyVersion!()).toEqual(first);
    expect(calls.filter((u) => u.includes("/api/version"))).toHaveLength(1);
  });

  it("takes a supplied version instead of asking", async () => {
    const { calls, fetchImpl } = versionFetch({ version_major: "26.1" });
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      serverVersion: "26.0",
      fetchImpl,
    });
    const report = await ctx.galaxyVersion!();
    expect(report.version).toMatchObject({ major: 26, minor: 0 });
    // Nothing was fetched, so there is no payload to report -- and no second opinion to differ.
    expect(report.payload).toBeUndefined();
    expect(calls).toEqual([]);
  });

  it("drops a failure as soon as it settles, so the next ask is a fresh one", async () => {
    // No window, no clock: nothing here keeps two callers consistent. A call that must not
    // change its mind pins one report instead (see the version guard tests).
    const { calls, fetchImpl } = versionFetch({ err_msg: "no" }, 500);
    const ctx = createGalaxyContext({ baseUrl: "https://g.example", apiKey: "K", fetchImpl });
    expect((await ctx.galaxyVersion!()).version).toBeUndefined();
    expect((await ctx.galaxyVersion!()).version).toBeUndefined();
    expect(calls.filter((u) => u.includes("/api/version"))).toHaveLength(2);
  });

  it("stops asking once the server recovers", async () => {
    const calls: string[] = [];
    let healthy = false;
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async (req: Request) => {
        calls.push(req.url);
        const body = healthy ? { version_major: "26.1" } : { err_msg: "no" };
        return new Response(JSON.stringify(body), {
          status: healthy ? 200 : 503,
          headers: { "content-type": "application/json" },
        });
      }) as unknown as typeof fetch,
    });
    expect((await ctx.galaxyVersion!()).version).toBeUndefined();
    healthy = true;
    expect((await ctx.galaxyVersion!()).version).toMatchObject({ major: 26, minor: 1 });
    expect((await ctx.galaxyVersion!()).version).toMatchObject({ major: 26, minor: 1 });
    expect(calls).toHaveLength(2); // an answer is kept for good; only a failure is dropped
  });

  it("gives up on a version route that never answers", async () => {
    // A hung /api/version must not become a hung everything. The probe carries its own deadline.
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      versionProbeTimeoutMs: 20,
      fetchImpl: hangingFetch(),
    });
    const report = await ctx.galaxyVersion!();
    expect(report.version).toBeUndefined();
    expect(report.source).toBe("unknown");
    expect(report.error?.kind).toBe("connection");
  });

  it("gives up when the caller cancels", async () => {
    const ac = new AbortController();
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      signal: ac.signal,
      fetchImpl: hangingFetch(),
    });
    const pending = ctx.galaxyVersion!();
    ac.abort();
    const report = await pending;
    expect(report.version).toBeUndefined();
    expect(report.source).toBe("unknown");
  });

  it("does not let one abandoned probe answer for the next call", async () => {
    let hang = true;
    const calls: string[] = [];
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      versionProbeTimeoutMs: 20,
      fetchImpl: (async (req: Request, init?: RequestInit) => {
        calls.push(req.url);
        if (hang) return hangingFetch()(req, init);
        return new Response(JSON.stringify({ version_major: "26.1" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }) as unknown as typeof fetch,
    });
    expect((await ctx.galaxyVersion!()).version).toBeUndefined();
    hang = false;
    expect((await ctx.galaxyVersion!()).version).toMatchObject({ major: 26, minor: 1 });
    expect(calls).toHaveLength(2);
  });

  it("falls through to the server when a supplied version is not a version", async () => {
    // An unset env var arrives as "", which must not switch the checking off for good.
    for (const junk of ["", "  ", "latest"]) {
      const { calls, fetchImpl } = versionFetch({ version_major: "26.1" });
      const ctx = createGalaxyContext({
        baseUrl: "https://g.example",
        apiKey: "K",
        serverVersion: junk,
        fetchImpl,
      });
      expect((await ctx.galaxyVersion!()).version).toMatchObject({ major: 26, minor: 1 });
      expect(calls.filter((u) => u.includes("/api/version"))).toHaveLength(1);
    }
  });

  it("is unknown when the server cannot be reached at all, but says why", async () => {
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async () => {
        throw new TypeError("fetch failed");
      }) as unknown as typeof fetch,
    });
    const report = await ctx.galaxyVersion!();
    expect(report.version).toBeUndefined();
    // The guard shrugs at this; get_server_info has to report it, so the reason is kept.
    expect(report.error?.kind).toBe("connection");
  });

  it("carries the status when the server refuses to answer", async () => {
    const { fetchImpl } = versionFetch({ err_msg: "nope" }, 401);
    const ctx = createGalaxyContext({ baseUrl: "https://g.example", apiKey: "K", fetchImpl });
    const report = await ctx.galaxyVersion!();
    expect(report.version).toBeUndefined();
    expect(report.error?.kind).toBe("auth");
  });

  it("is unknown when the payload carries no version_major", async () => {
    const { fetchImpl } = versionFetch({ version_minor: "1" });
    const ctx = createGalaxyContext({ baseUrl: "https://g.example", apiKey: "K", fetchImpl });
    const report = await ctx.galaxyVersion!();
    expect(report.version).toBeUndefined();
    expect(report.payload).toEqual({ version_minor: "1" }); // it answered, just not usefully
  });
});
