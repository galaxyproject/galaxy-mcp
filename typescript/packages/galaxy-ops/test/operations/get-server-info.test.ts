import { describe, it, expect } from "vitest";
import "../../src/operations/all"; // every op has to be registered for unsupported_tools to mean anything
import { getServerInfoOp, getServerInfo } from "../../src/operations/get-server-info";
import { allOperations, runWithEnvelope } from "../../src/operations/registry";
import { listPages } from "../../src/operations/list-pages";
import { createGalaxyContext, type GalaxyContext } from "../../src/context";
import { GalaxyVersionError } from "../../src/errors";

const gatedNames = allOperations.filter((op) => op.requires).map((op) => op.name);
if (gatedNames.length === 0) throw new Error("no op declares a requirement -- this file would prove nothing");

/**
 * A context over a stub server. `version` decides what /api/version answers on each call, so a
 * test can make the server change its mind between one request and the next.
 */
function serverThatSays(version: (nth: number) => { body: unknown; status: number }) {
  const paths: string[] = [];
  let nth = 0;
  const ctx = createGalaxyContext({
    baseUrl: "https://g.example",
    apiKey: "K",
    fetchImpl: (async (req: Request) => {
      const path = new URL(req.url).pathname;
      paths.push(path);
      const { body, status } =
        path === "/api/version"
          ? version(nth++)
          : { body: { brand: "Test" }, status: 200 };
      return new Response(JSON.stringify(body), {
        status,
        headers: { "content-type": "application/json" },
      });
    }) as unknown as typeof fetch,
  });
  return { ctx, paths };
}

const always = (versionMajor: unknown) => () => ({
  body: versionMajor === undefined ? {} : { version_major: versionMajor },
  status: 200,
});

describe("get_server_info", () => {
  it("has parity name and returns url + version + config", async () => {
    expect(getServerInfoOp.name).toBe("get_server_info");
    const { ctx } = serverThatSays(always("26.0"));
    const out = await getServerInfo({}, ctx);
    expect(out.url).toBe("https://g.example");
    expect((out.version as { version_major?: string }).version_major).toBe("26.0");
    expect((out.config as { brand?: string }).brand).toBe("Test");
  });

  it("names the ops a 26.0 server cannot run", async () => {
    const { ctx } = serverThatSays(always("26.0"));
    const out = await getServerInfo({}, ctx);
    expect(out.version_known).toBe(true);
    expect(out.unsupported_tools.map((o) => o.name).sort()).toEqual([...gatedNames].sort());
    expect(out.unsupported_tools.every((o) => o.requires === ">=26.1")).toBe(true);
  });

  it("lists nothing on a server new enough for everything", async () => {
    const { ctx } = serverThatSays(always("26.1"));
    const out = await getServerInfo({}, ctx);
    expect(out.version_known).toBe(true);
    expect(out.unsupported_tools).toEqual([]);
  });

  it("says the version is unknown rather than implying everything works", async () => {
    for (const reported of [undefined, "who knows"]) {
      const { ctx } = serverThatSays(always(reported));
      const out = await getServerInfo({}, ctx);
      expect(out.version_known).toBe(false);
      expect(out.unsupported_tools).toEqual([]);
    }
  });

  it("asks /api/version once, through the context, not once more for itself", async () => {
    const { ctx, paths } = serverThatSays(always("26.0"));
    await getServerInfo({}, ctx);
    expect(paths.filter((p) => p === "/api/version")).toHaveLength(1);
  });

  it("surfaces a refused version lookup instead of reporting an empty server", async () => {
    const { ctx } = serverThatSays(() => ({ body: { err_msg: "nope" }, status: 401 }));
    await expect(getServerInfo({}, ctx)).rejects.toMatchObject({ kind: "auth" });
  });

  it("names the address it was connected to, and leaves the version to data", async () => {
    // version_source is not a field of the wire answer -- the other server has no such key
    // -- and its sentence says nothing about the version either. The library result keeps
    // it, because a library caller is the only one who can have supplied a version.
    const { ctx } = serverThatSays(always("26.0"));
    const out = await runWithEnvelope(getServerInfoOp as never, {}, ctx);
    expect("version_source" in (out.data as object)).toBe(false);
    expect(out.message).toBe("Retrieved server info for https://g.example");
    const lib = await getServerInfoOp.run({}, ctx);
    expect(lib.version_source).toBe("server");
  });

  it("says unknown when nothing readable came back", async () => {
    const { ctx } = serverThatSays(always("who knows"));
    const out = await getServerInfo({}, ctx);
    expect(out.version_known).toBe(false);
  });

  it("reports a supplied version instead of throwing it away", async () => {
    // The context knows the version and the guard enforces it; answering "version ?" for one
    // we are actively refusing ops over is the worst of both.
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      serverVersion: "26.0",
      fetchImpl: (async () =>
        new Response(JSON.stringify({ brand: "Test" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        })) as unknown as typeof fetch,
    });
    const out = await runWithEnvelope(getServerInfoOp as never, {}, ctx);
    const info = out.data as { version_known: boolean; version: { version_major?: string } };
    expect(info.version_known).toBe(true);
    expect(info.version.version_major).toBe("26.0");
    // A supplied version must not read as one the server was asked for: the library result
    // says where it came from, and the wire -- where nobody could have supplied one -- omits it.
    const lib = await getServerInfoOp.run({}, ctx);
    expect(lib.version_source).toBe("supplied");
    expect("version_source" in (out.data as object)).toBe(false);
  });

  it("normalises a supplied full version to a version_major", async () => {
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      serverVersion: "26.1.1",
      fetchImpl: (async () =>
        new Response(JSON.stringify({ brand: "Test" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        })) as unknown as typeof fetch,
    });
    const out = await getServerInfo({}, ctx);
    expect((out.version as { version_major?: string }).version_major).toBe("26.1");
    expect(out.unsupported_tools).toEqual([]);
  });

  it("agrees with the guard when the caller overrode the version", async () => {
    // createGalaxyContext honours serverVersion, and so does the guard -- so must this list.
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      serverVersion: "26.0",
      fetchImpl: (async () =>
        new Response(JSON.stringify({ brand: "Test", version_major: "26.1" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        })) as unknown as typeof fetch,
    });
    const out = await getServerInfo({}, ctx);
    expect(out.unsupported_tools.map((o) => o.name).sort()).toEqual([...gatedNames].sort());
    await expect(listPages({}, ctx)).rejects.toBeInstanceOf(GalaxyVersionError);
  });
});

describe("get_server_info and the guard cannot reach different answers", () => {
  /** One good /api/version, then the server stops answering it. */
  const goodThen401 = (nth: number) =>
    nth === 0
      ? { body: { version_major: "26.0" }, status: 200 }
      : { body: { err_msg: "nope" }, status: 401 };

  it("keeps the one good answer for both, instead of one of them re-probing into a 401", async () => {
    const { ctx, paths } = serverThatSays(goodThen401);
    const out = await getServerInfo({}, ctx);
    expect(out.version_known).toBe(true);
    expect(out.unsupported_tools).toHaveLength(gatedNames.length);
    // The guard must not now decide the version is unknown and let a gated op through.
    await expect(listPages({ historyId: "H" }, ctx)).rejects.toBeInstanceOf(GalaxyVersionError);
    expect(paths.filter((p) => p === "/api/version")).toHaveLength(1);
    expect(paths).not.toContain("/api/pages");
  });

  it("lets the guard see an answer that get_server_info was the one to fetch", async () => {
    const { ctx, paths } = serverThatSays(goodThen401);
    await getServerInfo({}, ctx);
    await expect(listPages({}, ctx)).rejects.toBeInstanceOf(GalaxyVersionError);
    expect(paths.filter((p) => p === "/api/version")).toHaveLength(1);
  });

  it("works in the other order too: the guard fetches, get_server_info reuses", async () => {
    const { ctx, paths } = serverThatSays(goodThen401);
    await expect(listPages({}, ctx)).rejects.toBeInstanceOf(GalaxyVersionError);
    const out = await getServerInfo({}, ctx);
    expect(out.version_known).toBe(true);
    expect((out.version as { version_major?: string }).version_major).toBe("26.0");
    expect(paths.filter((p) => p === "/api/version")).toHaveLength(1);
  });
});

describe("get_server_info's summary line", () => {
  const summary = (over: Record<string, unknown>) =>
    getServerInfoOp.project?.(
      {
        url: "u",
        version: { version_major: "26.0" },
        config: {},
        version_known: true,
        unsupported_tools: [],
        ...over,
      } as never,
      {} as never,
    )?.message;

  it("is the address that was connected to, and nothing else", () => {
    // The other server's sentence names the url and stops. The version, whether it
    // could be read, and the tools this server is too old for are all in data, so
    // repeating any of them here would be one surface saying more than the other.
    expect(summary({})).toBe("Retrieved server info for u");
    expect(
      summary({ unsupported_tools: [{ name: "get_page_revision", requires: ">=26.1" }] }),
    ).toBe("Retrieved server info for u");
    expect(summary({ version: {}, version_known: false })).toBe("Retrieved server info for u");
  });

  it("says nothing about where the version came from, because data cannot either", async () => {
    // A supplied version is enforced by the guard and reported in data like any other;
    // that it was supplied rather than fetched is no longer anywhere in the envelope.
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      serverVersion: "26.0",
      fetchImpl: (async () =>
        new Response(JSON.stringify({ brand: "Test" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        })) as unknown as typeof fetch,
    });
    const out = await runWithEnvelope(getServerInfoOp as never, {}, ctx);
    expect(out.message).toBe("Retrieved server info for https://g.example");
    expect((out.data as { version: { version_major?: string } }).version.version_major).toBe("26.0");
  });
});

describe("a context with no version lookup at all", () => {
  it("reports the version as unknown rather than inventing a second opinion", async () => {
    // Hand-assembled contexts have no lookup; the guard treats them as unknown, so this must
    // too, or the two disagree again in the other direction.
    const ctx: GalaxyContext = {
      client: {
        GET: async () => ({ data: { brand: "Test" }, response: { status: 200 } }),
      } as unknown as GalaxyContext["client"],
      baseUrl: "https://g.example",
      poll: { intervalMs: 1, maxIntervalMs: 1, backoff: 1, jitter: 0, timeoutMs: 1 },
    };
    const out = await getServerInfo({}, ctx);
    expect(out.version_known).toBe(false);
    expect(out.unsupported_tools).toEqual([]);
  });
});
