/**
 * `whoami`, the live check of a token, against the fake Galaxy over a real socket: what it counts as
 * a user, how it tells a bad token from a bad address or a Galaxy that is down, and that nothing
 * reaches the key before the address is shown to be a Galaxy.
 */
import { beforeEach, describe, expect, it } from "vitest";
import { createWhoami, probeGalaxy } from "../../src/auth/whoami";
import { wasAnswered } from "../../src/auth/network";
import { startFakeGalaxy, useFakeGalaxies, type FakeGalaxy } from "../support/fake-galaxy";
import type { BaseUrl } from "../../src/auth/types";
import { version } from "../../package.json";
import { authErrorOf } from "./harness";

const start = useFakeGalaxies();
let galaxy: FakeGalaxy;
beforeEach(async () => {
  galaxy = await start();
});
const base = () => galaxy.baseUrl as BaseUrl;

const certificateError = () =>
  new TypeError("fetch failed", {
    cause: Object.assign(new Error("self-signed certificate"), {
      code: "DEPTH_ZERO_SELF_SIGNED_CERT",
    }),
  });
/** A `fetch` that fails every request the way a certificate failure does. */
const tlsFailure = (async () => {
  throw certificateError();
}) as unknown as typeof fetch;
/** The same, except that it answers the version probe, so the failure comes at the user lookup. */
const tlsFailureAfterProbe = (async (input: Request | string | URL) => {
  if (String(input instanceof Request ? input.url : input).endsWith("/api/version"))
    return Response.json({ version_major: "26.1" });
  throw certificateError();
}) as typeof fetch;

describe("whoami", () => {
  it("returns the user and whether they are an admin", async () => {
    expect(await createWhoami()(base(), galaxy.issueToken({ admin: true }))).toEqual({
      username: "fake-user",
      email: "fake-user@example.org",
      isAdmin: true,
    });
    expect((await createWhoami()(base(), galaxy.issueToken())).isAdmin).toBe(false);
  });

  it("refuses the anonymous record Galaxy answers with 200 for a key it does not know", async () => {
    // Observed on Galaxy 26.1: 200 and no `username`. A status code alone says nothing here, so
    // success means a record with a real user in it (whoami.ts).
    expect((await authErrorOf(createWhoami()(base(), "not-a-real-token"))).kind).toBe(
      "not_logged_in",
    );
  });

  // A refusal says the token is bad and a 5xx says Galaxy is down. The next two are neither, and are
  // named for what they are: otherwise a person re-creates a good key for nothing.
  it.each([
    ["a 403", 403, "not_logged_in"],
    ["a 401", 401, "not_logged_in"],
    ["a 503", 503, "network"],
  ] as const)("treats %s as %s", async (_what, status, kind) => {
    galaxy.setUsersCurrentStatus(status);
    expect((await authErrorOf(createWhoami()(base(), "fake-token-xyz"))).kind).toBe(kind);
  });

  it("calls a 404 a non-Galaxy address, not a bad token", async () => {
    galaxy.setUsersCurrentStatus(404);
    const e = await authErrorOf(createWhoami()(base(), "fake-token-xyz"));
    expect(e.kind).toBe("protocol");
    expect(e.message).toContain("does not look like a Galaxy API");
    expect(e.message).not.toMatch(/invalid|expired/);
  });

  it("calls a 429 rate limiting, to retry later, not a bad token", async () => {
    galaxy.setUsersCurrentStatus(429);
    const e = await authErrorOf(createWhoami()(base(), "fake-token-xyz"));
    expect(e.kind).toBe("network");
    expect(e.message).toMatch(/rate-limiting/);
  });

  it("stops when the signal is already aborted, sending no token", async () => {
    const controller = new AbortController();
    controller.abort(new Error("ctrl-c"));
    await expect(createWhoami()(base(), galaxy.issueToken(), controller.signal)).rejects.toThrow(
      "ctrl-c",
    );
    expect(galaxy.requestsTo("/api/users/current")).toHaveLength(0);
  });

  // The key travels in a header, so a client that followed a redirect would hand it to whatever origin
  // the redirect named.
  it("never gives the key to the origin a Galaxy redirects to", async () => {
    const elsewhere = await start();
    galaxy.setRedirect("/api/users/current", `${elsewhere.baseUrl}/api/users/current`);
    const e = await authErrorOf(createWhoami()(base(), "fake-token-xyz"));
    // A wrong address, not a refused key: the person is not sent to re-create a good one.
    expect(e.kind).toBe("protocol");
    expect(e.message).toContain("redirects (HTTP 302)");
    expect(e.message).not.toMatch(/did not accept|invalid|expired/);
    expect(elsewhere.requests).toEqual([]);
  });

  // The proxy hint is for a request that got no answer. A 5xx or a 429 is an answer, so `fail` is
  // told not to offer it (`wasAnswered`); a failure to connect is not marked.
  it("marks a 5xx and a 429 as answered by the server, and a connection failure not", async () => {
    for (const status of [503, 429]) {
      galaxy.setUsersCurrentStatus(status);
      const e = await authErrorOf(createWhoami()(base(), "fake-token-xyz"));
      expect([e.kind, wasAnswered(e)], `HTTP ${status}`).toEqual(["network", true]);
    }
    const down = await authErrorOf(createWhoami(tlsFailure)(base(), "fake-token-xyz"));
    expect([down.kind, wasAnswered(down)]).toEqual(["network", false]);
  });
});

/** Answers the version probe, and answers the user lookup never, until its request is aborted. */
function stallingAfterProbe() {
  const lookups: AbortSignal[] = [];
  const fetchImpl = ((input: Request | string | URL, init?: RequestInit) => {
    const req = input instanceof Request ? input : new Request(input, init);
    if (req.url.endsWith("/api/version"))
      return Promise.resolve(Response.json({ version_major: "26.1" }));
    lookups.push(req.signal);
    return new Promise<Response>((_, reject) => {
      req.signal.addEventListener("abort", () => {
        reject(req.signal.reason as Error);
      });
    });
  }) as typeof fetch;
  return { fetchImpl, lookups };
}

// A Galaxy or a proxy that accepts the connection and then says nothing must not hang a script
// that nothing will Ctrl-C.
describe("a request that is never answered", () => {
  it("gives up on the user lookup at the timeout, as a network error", async () => {
    const { fetchImpl, lookups } = stallingAfterProbe();
    const e = await authErrorOf(createWhoami(fetchImpl, 50)(base(), "fake-token-xyz"));
    expect(e.kind).toBe("network");
    expect(e.message).toMatch(/Could not reach/);
    expect(lookups).toHaveLength(1);
  });

  it("gives up on the version probe at the timeout too", async () => {
    const hang = ((_input: Request | string | URL, init?: RequestInit) =>
      new Promise<Response>((_, reject) => {
        init?.signal?.addEventListener("abort", () => {
          reject(init.signal!.reason as Error);
        });
      })) as typeof fetch;
    const e = await authErrorOf(createWhoami(hang, 50)(base(), "fake-token-xyz"));
    expect(e.kind).toBe("network");
  });
});

// The probe is shared by every call for one Galaxy. A call that is cancelled stops waiting, and
// when none is left the probe's request is aborted: it would otherwise hold the process open
// after Ctrl-C until its own timeout.
describe("cancelling while the version probe is in flight", () => {
  /** A `fetch` whose version probe hangs until aborted, recording the signal it was given. */
  function hangingProbe() {
    const probes: AbortSignal[] = [];
    const fetchImpl = ((input: Request | string | URL, init?: RequestInit) => {
      const req = input instanceof Request ? input : new Request(input, init);
      if (!req.url.endsWith("/api/version"))
        return Promise.resolve(new Response("{}", { status: 500 }));
      probes.push(init?.signal ?? req.signal);
      return new Promise<Response>((_, reject) => {
        (init?.signal ?? req.signal).addEventListener("abort", () => {
          reject((init?.signal ?? req.signal).reason as Error);
        });
      });
    }) as typeof fetch;
    return { fetchImpl, probes };
  }
  const tick = () => new Promise((r) => setTimeout(r, 10));

  it("aborts the probe's request when the only caller is cancelled", async () => {
    const { fetchImpl, probes } = hangingProbe();
    const ac = new AbortController();
    const p = createWhoami(fetchImpl)(base(), "fake-token-xyz", ac.signal).catch((x: unknown) => x);
    await tick();
    expect(probes).toHaveLength(1);
    expect(probes[0]!.aborted).toBe(false);
    ac.abort(new Error("ctrl-c"));
    expect(((await p) as Error).message).toBe("ctrl-c");
    expect(probes[0]!.aborted).toBe(true);
  });

  it("keeps the probe going while another caller still waits on it, and aborts it when none does", async () => {
    const { fetchImpl, probes } = hangingProbe();
    const who = createWhoami(fetchImpl);
    const first = new AbortController();
    const second = new AbortController();
    const a = who(base(), "fake-token-xyz", first.signal).catch((x: unknown) => x);
    const b = who(base(), "fake-token-xyz", second.signal).catch((x: unknown) => x);
    await tick();
    first.abort(new Error("first"));
    expect(((await a) as Error).message).toBe("first");
    expect(probes).toHaveLength(1);
    expect(probes[0]!.aborted, "the second caller is still waiting").toBe(false);
    second.abort(new Error("second"));
    expect(((await b) as Error).message).toBe("second");
    expect(probes[0]!.aborted).toBe(true);
  });

  it("sends a new probe for a call made after the last one was cancelled", async () => {
    const { fetchImpl, probes } = hangingProbe();
    const who = createWhoami(fetchImpl);
    const first = new AbortController();
    const a = who(base(), "fake-token-xyz", first.signal).catch((x: unknown) => x);
    await tick();
    first.abort(new Error("first"));
    await a;
    const second = new AbortController();
    const b = who(base(), "fake-token-xyz", second.signal).catch((x: unknown) => x);
    await tick();
    expect(probes).toHaveLength(2);
    expect(probes[1]!.aborted).toBe(false);
    second.abort(new Error("second"));
    await b;
  });
});

// Before any key is sent, an anonymous `GET /api/version` shows the address is a Galaxy's API. A wrong
// address often redirects (http to https, a missing path prefix), and a Galaxy behind a `remote_user`
// proxy answers an HTML login page: sending the key along would hand it to whatever answers. Each
// refusal here also shows that no `/api/users/current` request, which carries the key, was made.
describe("version probe before the token is sent", () => {
  it("runs once per Galaxy, anonymously", async () => {
    const token = galaxy.issueToken();
    const who = createWhoami();
    await who(base(), token);
    await who(base(), token);
    const probes = galaxy.requestsTo("/api/version");
    expect(probes).toHaveLength(1);
    expect(probes[0]!.headers["x-api-key"]).toBeUndefined();
  });

  it("stops at a redirect, offering the address it points to, and sends no token", async () => {
    galaxy.setVersionMode("redirect");
    const e = await authErrorOf(createWhoami()(base(), "fake-token-xyz"));
    expect(e.kind).toBe("protocol");
    expect(e.message).toContain("https://elsewhere.example");
    expect(galaxy.requestsTo("/api/users/current")).toHaveLength(0);
  });

  it("stops at an answer that is not a Galaxy version, and sends no token", async () => {
    galaxy.setVersionMode("html");
    const e = await authErrorOf(createWhoami()(base(), "fake-token-xyz"));
    expect(e.kind).toBe("protocol");
    expect(e.message).toContain("does not look like a Galaxy API");
    expect(galaxy.requestsTo("/api/users/current")).toHaveLength(0);
  });
});

describe("network and TLS failures", () => {
  it("an unreachable host is a network error naming the CA setting and not the proxy", async () => {
    const dead = await startFakeGalaxy();
    const url = dead.baseUrl as BaseUrl;
    await dead.close();
    const e = await authErrorOf(probeGalaxy(url));
    expect(e.kind).toBe("network");
    expect(e.message).not.toContain("NODE_USE_ENV_PROXY");
    expect(e.message).toContain("NODE_EXTRA_CA_CERTS");
  });

  it("a certificate failure in the version probe names NODE_EXTRA_CA_CERTS", async () => {
    const e = await authErrorOf(probeGalaxy("https://galaxy.example.org" as BaseUrl, tlsFailure));
    expect(e.kind).toBe("network");
    expect(e.message).toContain("NODE_EXTRA_CA_CERTS");
    expect(e.message).not.toContain("NODE_USE_ENV_PROXY");
  });

  it("a certificate failure in the user lookup names NODE_EXTRA_CA_CERTS", async () => {
    const e = await authErrorOf(
      createWhoami(tlsFailureAfterProbe)("https://galaxy.example.org" as BaseUrl, "fake-token-xyz"),
    );
    expect(e.kind).toBe("network");
    expect(e.message).toContain("self-signed certificate");
    expect(e.message).toContain("NODE_EXTRA_CA_CERTS");
    expect(e.message).not.toContain("NODE_USE_ENV_PROXY");
  });
});

// So a Galaxy admin can tell `gxy` from other clients in their logs (user-agent.ts). The expected value is
// built from package.json, not from the constant under test, so a change to the string's shape fails.
describe("User-Agent", () => {
  it("is gxy/<package version> (galaxy-cli) on whoami and the probe", async () => {
    const token = galaxy.issueToken();
    await createWhoami()(base(), token);

    const paths = galaxy.requests.map((r) => new URL(r.url, "http://x").pathname);
    for (const expected of ["/api/version", "/api/users/current"]) {
      expect(paths).toContain(expected);
    }
    for (const r of galaxy.requests)
      expect(r.headers["user-agent"], r.url).toBe(`gxy/${version} (galaxy-cli)`);
  });
});
