import { describe, it, expect } from "vitest";
import { z } from "zod";
import {
  allOperations,
  register,
  runOperation,
  runWithEnvelope,
  describeOperation,
} from "../../src/operations/registry";
import type { AnyOperation, Operation } from "../../src/operations/types";
import { createGalaxyContext, type GalaxyContext } from "../../src/context";
import { GalaxyVersionError } from "../../src/errors";
import { parseGalaxyVersion } from "../../src/version";
import { mockClient } from "../util/mock-client";
import { listPagesOp, listPages } from "../../src/operations/list-pages";

const input = { id: z.string() };

let fixtureSeq = 0;

/** Counts what run() would send, so a refusal can be shown to have sent nothing. */
function newOp(requires?: { galaxy: string }, name = `needs_new_galaxy_${++fixtureSeq}`) {
  const sent: string[] = [];
  const op: Operation<typeof input, { ran: string }> = {
    name,
    domain: "pages",
    summary: "An op that wants a newer server.",
    input,
    ...(requires ? { requires } : {}),
    run: async (i) => {
      sent.push(i.id);
      return { ran: i.id };
    },
  };
  return { op, sent };
}

/** A context that reports a version without a server behind it. */
function ctxAt(version: string | undefined): GalaxyContext {
  return {
    client: mockClient({}),
    poll: { intervalMs: 1, maxIntervalMs: 1, backoff: 1, jitter: 0, timeoutMs: 1 },
    ...(version === undefined
      ? {}
      : {
          galaxyVersion: async () => ({
            version: parseGalaxyVersion(version),
            source: "server" as const,
          }),
        }),
  };
}

describe("the version guard", () => {
  it("refuses an op the server is too old for, and sends nothing", async () => {
    const { op, sent } = newOp({ galaxy: ">=26.1" }, "needs_new_galaxy");
    await expect(runOperation(op, { id: "x" }, ctxAt("26.0"))).rejects.toThrow(
      "needs_new_galaxy needs Galaxy 26.1 or newer; this server reports 26.0",
    );
    expect(sent).toEqual([]);
  });

  it("allows an equal or newer server", async () => {
    for (const version of ["26.1", "26.2", "27.0", "26.1.1"]) {
      const { op, sent } = newOp({ galaxy: ">=26.1" });
      await expect(runOperation(op, { id: version }, ctxAt(version))).resolves.toEqual({
        ran: version,
      });
      expect(sent).toEqual([version]);
    }
  });

  it("allows an unknown version rather than guessing", async () => {
    for (const ctx of [ctxAt(undefined), ctxAt("not a version")]) {
      const { op, sent } = newOp({ galaxy: ">=26.1" });
      await expect(runOperation(op, { id: "x" }, ctx)).resolves.toEqual({ ran: "x" });
      expect(sent).toEqual(["x"]);
    }
  });

  it("does not ask the server its version for an op that declares nothing", async () => {
    let asked = 0;
    const ctx: GalaxyContext = {
      ...ctxAt("26.0"),
      galaxyVersion: async () => {
        asked += 1;
        return { version: parseGalaxyVersion("26.0"), source: "server" as const };
      },
    };
    const { op } = newOp();
    await runOperation(op, { id: "x" }, ctx);
    expect(asked).toBe(0);
  });

  it("reaches a surface as a version error rather than a failed call", async () => {
    const { op, sent } = newOp({ galaxy: ">=26.1" });
    const result = await runWithEnvelope(op, { id: "x" }, ctxAt("26.0"));
    expect(result.success).toBe(false);
    expect(result.errorKind).toBe("version");
    expect(result.message).toContain("Galaxy 26.1 or newer");
    expect(sent).toEqual([]);
  });

  it("refuses a malformed requirement where the op registers, not where it runs", () => {
    const { op } = newOp({ galaxy: "26.1" });
    expect(() => register(op as AnyOperation)).toThrow(/">=MAJOR\.MINOR"/);
  });

  it("refuses to register the same op twice, which would wrap the wrapper", () => {
    const { op } = newOp({ galaxy: ">=26.1" });
    register(op as AnyOperation);
    try {
      expect(() => register(op as AnyOperation)).toThrow(/already registered/);
      const twin = { ...newOp({ galaxy: ">=26.1" }, op.name).op };
      expect(() => register(twin as AnyOperation)).toThrow(/already registered/);
    } finally {
      allOperations.splice(allOperations.indexOf(op as AnyOperation), 1);
    }
  });
});

describe("a registered op's own run", () => {
  it("carries the guard, so the exported object is not a way around it", async () => {
    const { op, sent } = newOp({ galaxy: ">=26.1" });
    register(op as AnyOperation);
    try {
      // Reaching past the named export straight at the object the registry handed out.
      await expect(op.run({ id: "x" }, ctxAt("26.0"))).rejects.toBeInstanceOf(GalaxyVersionError);
      expect(sent).toEqual([]);
      await expect(op.run({ id: "x" }, ctxAt("26.1"))).resolves.toEqual({ ran: "x" });
      expect(sent).toEqual(["x"]);
    } finally {
      allOperations.splice(allOperations.indexOf(op as AnyOperation), 1);
    }
  });

  it("reaches the server once per call however many guards run", async () => {
    // What matters is requests on the wire, not how many layers consulted the memo. The old
    // version of this test counted callbacks, which wrote the mechanism down as the requirement.
    const { op } = newOp({ galaxy: ">=26.1" });
    register(op as AnyOperation);
    const seen: string[] = [];
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async (req: Request) => {
        seen.push(new URL(req.url).pathname);
        return new Response(JSON.stringify({ version_major: "26.1" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }) as unknown as typeof fetch,
    });
    try {
      await runWithEnvelope(op, { id: "x" }, ctx);
      expect(seen.filter((p) => p === "/api/version")).toHaveLength(1);
      await op.run({ id: "x" }, ctx);
      expect(seen.filter((p) => p === "/api/version")).toHaveLength(1); // answer is memoized
    } finally {
      allOperations.splice(allOperations.indexOf(op as AnyOperation), 1);
    }
  });
});

describe("describeOperation", () => {
  it("appends the requirement so no op writes the sentence itself", () => {
    const { op } = newOp({ galaxy: ">=26.1" });
    expect(describeOperation(op)).toBe(
      "An op that wants a newer server. Requires Galaxy 26.1 or newer.",
    );
  });

  it("leaves an op that needs nothing alone", () => {
    const { op } = newOp();
    expect(describeOperation(op)).toBe("An op that wants a newer server.");
  });
});

describe("a real context", () => {
  it("refuses without sending the op's own request", async () => {
    const seen: string[] = [];
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async (req: Request) => {
        seen.push(new URL(req.url).pathname);
        return new Response(JSON.stringify({ version_major: "26.0" }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }) as unknown as typeof fetch,
    });
    const { op } = newOp({ galaxy: ">=26.1" });
    await expect(runOperation(op, { id: "x" }, ctx)).rejects.toThrow(/26\.1 or newer/);
    expect(seen).toEqual(["/api/version"]);
  });
});

describe("a call that cannot learn the version", () => {
  /**
   * A server that refuses the first ask and answers the second. Any call that probes twice
   * reaches two different decisions; a call that probes once cannot.
   */
  function flakyServer() {
    const paths: string[] = [];
    let asks = 0;
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async (req: Request) => {
        const path = new URL(req.url).pathname;
        paths.push(path);
        if (path !== "/api/version") {
          return new Response("[]", {
            status: 200,
            headers: { "content-type": "application/json" },
          });
        }
        const refused = asks++ === 0;
        return new Response(
          JSON.stringify(refused ? { err_msg: "no" } : { version_major: "26.0" }),
          { status: refused ? 401 : 200, headers: { "content-type": "application/json" } },
        );
      }) as unknown as typeof fetch,
    });
    return { ctx, paths };
  }

  const entryPoints: [string, (ctx: GalaxyContext) => Promise<unknown>][] = [
    [
      "runWithEnvelope, which is what the CLI and MCP use",
      (ctx) => runWithEnvelope(listPagesOp as never, {} as never, ctx),
    ],
    ["the named export", (ctx) => listPages({}, ctx)],
    ["the op object's own run", (ctx) => listPagesOp.run({}, ctx)],
  ];

  it.each(entryPoints)("probes once and proceeds on an unknown version: %s", async (_name, call) => {
    const { ctx, paths } = flakyServer();
    await call(ctx);
    expect(paths.filter((p) => p === "/api/version")).toHaveLength(1);
    // Unknown refuses nothing, so the op ran. A second probe would have seen 26.0 and refused.
    expect(paths).toContain("/api/pages");
  });

  it("reaches the same decision whichever entry point was used", async () => {
    const outcomes = [];
    for (const [, call] of entryPoints) {
      const { ctx, paths } = flakyServer();
      let refused = false;
      try {
        const envelope = (await call(ctx)) as { success?: boolean; errorKind?: string };
        refused = envelope?.success === false && envelope.errorKind === "version";
      } catch (err) {
        refused = err instanceof GalaxyVersionError;
      }
      outcomes.push({
        refused,
        probes: paths.filter((p) => p === "/api/version").length,
        ran: paths.includes("/api/pages"),
      });
    }
    expect(outcomes).toEqual([
      { refused: false, probes: 1, ran: true },
      { refused: false, probes: 1, ran: true },
      { refused: false, probes: 1, ran: true },
    ]);
  });

  it("makes one request and reaches one answer even though two guards run", async () => {
    // The call goes through runOperation's guard and then the wrapper's. The server would hand
    // a second probe a different answer, so if either guard re-read the version they would
    // disagree. Pinning the first reading onto the context makes that impossible.
    const { ctx, paths } = flakyServer();
    await listPages({}, ctx);
    expect(paths.filter((p) => p === "/api/version")).toHaveLength(1);
    expect(paths).toContain("/api/pages"); // 401 -> unknown -> both guards let it through
  });

  it("acts on what the next call learns, because a failure is not kept", async () => {
    const { ctx, paths } = flakyServer();
    await listPages({}, ctx); // 401 -> unknown -> proceeds
    await expect(listPages({}, ctx)).rejects.toBeInstanceOf(GalaxyVersionError); // now 26.0
    await expect(listPages({}, ctx)).rejects.toBeInstanceOf(GalaxyVersionError); // answer kept
    expect(paths.filter((p) => p === "/api/version")).toHaveLength(2);
  });
});

describe("an op object someone assembled themselves", () => {
  /** 26.1: new enough for the Pages ops as registered, too old for a stricter copy. */
  function serverAt(versionMajor: string) {
    const paths: string[] = [];
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      fetchImpl: (async (req: Request) => {
        const path = new URL(req.url).pathname;
        paths.push(path);
        const body = path === "/api/version" ? { version_major: versionMajor } : [];
        return new Response(JSON.stringify(body), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }) as unknown as typeof fetch,
    });
    return { ctx, paths };
  }

  const refusedBy = async (call: () => Promise<unknown>) => {
    try {
      const envelope = (await call()) as { success?: boolean; errorKind?: string };
      return envelope?.success === false && envelope.errorKind === "version";
    } catch (err) {
      return err instanceof GalaxyVersionError;
    }
  };

  it("is held to its own stricter requirement, not the one it was copied from", async () => {
    // The copy wants 27.0; the op it was spread from wants 26.1 and the server is 26.1.
    const stricter = { ...listPagesOp, requires: { galaxy: ">=27.0" } };
    for (const call of [
      (ctx: GalaxyContext) => runWithEnvelope(stricter as never, {} as never, ctx),
      (ctx: GalaxyContext) => runOperation(stricter as never, {} as never, ctx),
      (ctx: GalaxyContext) => stricter.run({}, ctx),
    ]) {
      const { ctx, paths } = serverAt("26.1");
      expect(await refusedBy(() => call(ctx))).toBe(true);
      expect(paths).not.toContain("/api/pages");
    }
  });

  it("cannot loosen the requirement it was copied from", async () => {
    // The copy asks for 26.0 and the server is 26.0, but list_pages needs 26.1 -- and on 26.0
    // its history filter is silently ignored, which is the whole reason it is gated.
    const weaker = { ...listPagesOp, requires: { galaxy: ">=26.0" } };
    for (const call of [
      (ctx: GalaxyContext) => runWithEnvelope(weaker as never, { historyId: "H" } as never, ctx),
      (ctx: GalaxyContext) => runOperation(weaker as never, { historyId: "H" } as never, ctx),
      (ctx: GalaxyContext) => weaker.run({ historyId: "H" }, ctx),
      (ctx: GalaxyContext) =>
        ({ ...listPagesOp, requires: { galaxy: ">=26.0" }, run: listPagesOp.run.bind(listPagesOp) })
          .run({ historyId: "H" }, ctx),
    ]) {
      const { ctx, paths } = serverAt("26.0");
      expect(await refusedBy(() => call(ctx))).toBe(true);
      expect(paths).not.toContain("/api/pages");
    }
  });

  it("cannot edit the requirement out from under the op it was copied from", async () => {
    // A shallow copy shares the nested requires object, so writing through it used to rewrite
    // what the original enforces -- for every later caller, not just this one.
    const copy = { ...listPagesOp };
    expect(copy.requires).toBe(listPagesOp.requires); // the sharing is real, not hypothetical
    expect(() => {
      copy.requires!.galaxy = ">=26.0";
    }).toThrow(TypeError);
    expect(listPagesOp.requires?.galaxy).toBe(">=26.1");

    for (const call of [
      (ctx: GalaxyContext) => runWithEnvelope(copy as never, { historyId: "H" } as never, ctx),
      (ctx: GalaxyContext) => runOperation(copy as never, { historyId: "H" } as never, ctx),
      (ctx: GalaxyContext) => copy.run({ historyId: "H" }, ctx),
      // and the original, which is the one that was being weakened
      (ctx: GalaxyContext) => listPages({ historyId: "H" }, ctx),
      (ctx: GalaxyContext) => listPagesOp.run({ historyId: "H" }, ctx),
    ]) {
      const { ctx, paths } = serverAt("26.0");
      expect(await refusedBy(() => call(ctx))).toBe(true);
      expect(paths).not.toContain("/api/pages");
    }
  });

  it("cannot have any of its parts swapped out after registration", async () => {
    // A registered op is sealed. Until it was, `listPagesOp.run = mine` quietly removed the
    // guard and left a direct call less protected than every other way in, and swapping
    // `requires` let the op advertise one version while enforcing another.
    const sends: string[] = [];
    const replacements: [string, () => void][] = [
      [
        "run",
        () => {
          (listPagesOp as { run: unknown }).run = async (_i: unknown, c: GalaxyContext) => {
            sends.push("ran");
            await c.client.GET("/api/pages", {});
            return [];
          };
        },
      ],
      ["requires", () => ((listPagesOp as { requires?: unknown }).requires = { galaxy: ">=26.0" })],
      ["name", () => ((listPagesOp as { name: string }).name = "my_pages")],
      ["summary", () => ((listPagesOp as { summary: string }).summary = "anything")],
      ["input", () => ((listPagesOp as { input: unknown }).input = {})],
    ];
    for (const [what, replace] of replacements) {
      expect(replace, `${what} was replaceable`).toThrow(TypeError);
    }
    expect(sends).toEqual([]);
    expect(listPagesOp.name).toBe("list_pages");
    expect(listPagesOp.requires?.galaxy).toBe(">=26.1");

    // and the direct call is still the guarded one
    for (const call of [
      (ctx: GalaxyContext) => runWithEnvelope(listPagesOp as never, { historyId: "H" } as never, ctx),
      (ctx: GalaxyContext) => listPages({ historyId: "H" }, ctx),
      (ctx: GalaxyContext) => listPagesOp.run({ historyId: "H" }, ctx),
    ]) {
      const { ctx, paths } = serverAt("26.0");
      expect(await refusedBy(() => call(ctx))).toBe(true);
      expect(paths).not.toContain("/api/pages");
    }
  });

  it("is blamed by the name the caller used, whichever way in they came", async () => {
    const alias = { ...listPagesOp, name: "my_pages" };
    const messageFrom = async (call: () => Promise<unknown>) => {
      try {
        const envelope = (await call()) as { message?: string };
        return envelope?.message ?? "";
      } catch (err) {
        return (err as Error).message;
      }
    };
    const { ctx } = serverAt("26.0");
    const viaEnvelope = await messageFrom(() => runWithEnvelope(alias as never, {} as never, ctx));
    const viaRun = await messageFrom(() => alias.run({}, ctx));
    expect(viaRun).toContain("my_pages");
    expect(viaRun).toBe(viaEnvelope);
  });

  it("still gets the registered requirement when the copy dropped it", async () => {
    // Never less guarded than the op it was made from.
    const { requires: _dropped, ...without } = listPagesOp;
    const { ctx, paths } = serverAt("26.0");
    expect(await refusedBy(() => (without as typeof listPagesOp).run({}, ctx))).toBe(true);
    expect(paths).not.toContain("/api/pages");
  });

  it("decides the same as the original when its run was rebound", async () => {
    // 401 then 26.0: a path that probes twice sees both and refuses; one that probes once does
    // not. The two must not disagree, however many guards each happens to run.
    function flaky() {
      const paths: string[] = [];
      let asks = 0;
      const ctx = createGalaxyContext({
        baseUrl: "https://g.example",
        apiKey: "K",
        fetchImpl: (async (req: Request) => {
          const path = new URL(req.url).pathname;
          paths.push(path);
          if (path !== "/api/version") {
            return new Response("[]", {
              status: 200,
              headers: { "content-type": "application/json" },
            });
          }
          const refused = asks++ === 0;
          return new Response(
            JSON.stringify(refused ? { err_msg: "no" } : { version_major: "26.0" }),
            { status: refused ? 401 : 200, headers: { "content-type": "application/json" } },
          );
        }) as unknown as typeof fetch,
      });
      return { ctx, paths };
    }

    const rebound = { ...listPagesOp, run: listPagesOp.run.bind(listPagesOp) };
    const outcomes = [];
    for (const op of [listPagesOp, rebound]) {
      const { ctx, paths } = flaky();
      outcomes.push({
        refused: await refusedBy(() => runWithEnvelope(op as never, {} as never, ctx)),
        probes: paths.filter((p) => p === "/api/version").length,
        ran: paths.includes("/api/pages"),
      });
    }
    expect(outcomes[1]).toEqual(outcomes[0]);
    expect(outcomes[0]).toEqual({ refused: false, probes: 1, ran: true });
  });
});

describe("a version route that never answers", () => {
  /** /api/version hangs until abandoned; /api/pages works perfectly well. */
  function hungVersion(signal?: AbortSignal) {
    const paths: string[] = [];
    const ctx = createGalaxyContext({
      baseUrl: "https://g.example",
      apiKey: "K",
      versionProbeTimeoutMs: 20,
      ...(signal ? { signal } : {}),
      fetchImpl: (async (req: Request, init?: RequestInit) => {
        const path = new URL(req.url).pathname;
        paths.push(path);
        if (path !== "/api/version") {
          return new Response("[]", {
            status: 200,
            headers: { "content-type": "application/json" },
          });
        }
        const abort = init?.signal ?? req.signal;
        await new Promise((_resolve, reject) => {
          if (abort?.aborted) return reject(abort.reason);
          abort?.addEventListener("abort", () => reject(abort.reason), { once: true });
        });
        throw new Error("unreachable");
      }) as unknown as typeof fetch,
    });
    return { ctx, paths };
  }

  it("does not stop a gated op whose own endpoint is fine", async () => {
    const { ctx, paths } = hungVersion();
    await expect(listPages({}, ctx)).resolves.toEqual([]);
    expect(paths).toContain("/api/pages"); // unknown version refuses nothing
  });

  it("is released when the caller cancels instead of holding the call open", async () => {
    const ac = new AbortController();
    const { ctx, paths } = hungVersion(ac.signal);
    const call = listPages({}, ctx);
    ac.abort();
    await expect(call).resolves.toEqual([]);
    expect(paths).toContain("/api/pages");
  });

  it("leaves the next call free to ask again", async () => {
    const { ctx } = hungVersion();
    await listPages({}, ctx);
    await expect(listPages({}, ctx)).resolves.toEqual([]);
  });
});
