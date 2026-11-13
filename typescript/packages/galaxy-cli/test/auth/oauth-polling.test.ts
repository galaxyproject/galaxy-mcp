/**
 * How the device login waits: the interval and `slow_down` (RFC 8628 §3.5), transient
 * failures, Retry-After, the local deadline and cancelling.
 *
 * Timing tests fake only `setTimeout` and `clearTimeout`, so the sockets still run for real.
 * They do not guess how long the sockets need: the client's `fetch` is wrapped to know when
 * a request has been answered in full, and after each step of the fake clock `tick` waits for
 * exactly that. What remains after it is in-process work with no I/O left to wait for.
 *
 * `setImmediate` is left real: `until` needs real event-loop turns to see the sockets finish.
 * The client's own deadline is a plain `setTimeout`, not `AbortSignal.timeout`, so that these
 * fake timers can move it (oauth.ts).
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { discover as discoverWith, type DiscoverOptions } from "../../src/auth/oauth";
import type { BaseUrl, PendingDeviceLogin } from "../../src/auth/types";
import { abortReason } from "../../src/auth/abort";
import { useFakeGalaxies, type FakeGalaxy } from "../support/fake-galaxy";
import { authErrorOf } from "./harness";

/** Discovery with no floor on the poll interval: the fake asks for 10 ms, and the real floor is 1 s. */
const discover = (baseUrl: BaseUrl, options: DiscoverOptions = {}) =>
  discoverWith(baseUrl, { minIntervalMs: 0, ...options });

const start = useFakeGalaxies();
let galaxy: FakeGalaxy;
const url = () => galaxy.baseUrl as BaseUrl;
const tokenRequests = () => galaxy.requestsTo("/oauth/token").length;

afterEach(() => {
  vi.useRealTimers();
});

/** The URL a `fetch` was given, whichever way it was given. */
const urlOf = (input: Parameters<typeof fetch>[0]): string =>
  input instanceof Request ? input.url : input.toString();

/** Real event-loop turns until `condition` holds, or fail after a generous bound. */
async function until(condition: () => boolean, what: string): Promise<void> {
  const deadline = performance.now() + 10_000;
  while (!condition()) {
    if (performance.now() > deadline) throw new Error(`timed out waiting for ${what}`);
    await new Promise((resolve) => setImmediate(resolve));
  }
}
const turn = () => new Promise((resolve) => setImmediate(resolve));

/** A `fetch` that knows how many requests are still waiting for their answer (body included). */
function tracked() {
  let inflight = 0;
  const impl: typeof fetch = async (input, init) => {
    inflight += 1;
    try {
      const res = await fetch(input, init);
      await res.clone().arrayBuffer();
      return res;
    } finally {
      inflight -= 1;
    }
  };
  return { fetch: impl, idle: () => inflight === 0 };
}

/** Discovery through a tracked `fetch`. */
async function connect() {
  const net = tracked();
  return { server: await discover(url(), { fetch: net.fetch }), net };
}

/** Slack on top of an interval, so a step lands just after the timer it means to fire. */
const JUST_AFTER = 100;

/** Start waiting under fake timers. `tick(ms)` moves the clock, then waits for any poll it fired to be answered. */
function startWaiting(
  pending: PendingDeviceLogin,
  net: { idle: () => boolean },
  signal?: AbortSignal,
) {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  const done = pending.waitForApproval(signal);
  return {
    done,
    async tick(ms: number) {
      await vi.advanceTimersByTimeAsync(ms);
      await until(net.idle, "the poll to be answered");
      await turn();
      await turn();
    },
  };
}

describe("polling", () => {
  beforeEach(async () => {
    galaxy = await start({ interval: 0.01 });
  });

  it("keeps polling while pending, then succeeds once approved", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    const done = pending.waitForApproval();
    await vi.waitFor(() => {
      expect(tokenRequests()).toBeGreaterThanOrEqual(3);
    });
    galaxy.approve(pending.userCode, { scopes: ["read"] });
    await expect(done).resolves.toMatchObject({ scopes: ["read"] });
  });

  // RFC 8628 §3.2: `interval` is the least the client waits between polls to the token endpoint.
  it("waits the server's interval before the first poll and between polls", async () => {
    galaxy = await start({ interval: 2 });
    const { server, net } = await connect();
    const w = startWaiting(await server.startDeviceLogin(), net);
    w.done.catch(() => undefined);
    const pending = galaxy.pendingCodes()[0]!;

    await w.tick(2000 - JUST_AFTER);
    expect(tokenRequests()).toBe(0);
    await w.tick(2 * JUST_AFTER);
    expect(tokenRequests()).toBe(1);
    await w.tick(2000 - 2 * JUST_AFTER);
    expect(tokenRequests()).toBe(1);
    await w.tick(2 * JUST_AFTER);
    expect(tokenRequests()).toBe(2);

    galaxy.approve(pending);
    await vi.advanceTimersByTimeAsync(2000 + JUST_AFTER);
    await expect(w.done).resolves.toMatchObject({ scopes: [] });
  });

  it("adds 5 seconds to the interval after slow_down", async () => {
    galaxy = await start({ interval: 1 });
    const { server, net } = await connect();
    galaxy.slowDown(1);
    const w = startWaiting(await server.startDeviceLogin(), net);
    w.done.catch(() => undefined);

    await w.tick(1000 + JUST_AFTER); // first poll: slow_down
    expect(tokenRequests()).toBe(1);
    // The interval is now 6 s: not yet. It stays 6 s from here on (the last wait is 6 s too), because
    // slow_down sets a new interval rather than delaying one poll.
    await w.tick(6000 - 2 * JUST_AFTER);
    expect(tokenRequests()).toBe(1);
    await w.tick(3 * JUST_AFTER);
    expect(tokenRequests()).toBe(2);

    galaxy.approve(galaxy.pendingCodes()[0]!);
    await vi.advanceTimersByTimeAsync(6000 + JUST_AFTER);
    await expect(w.done).resolves.toBeDefined();
  });
});

// Giving openid-client a signal replaces its own `expires_in` deadline (oauth.ts), so the deadline
// is the client's, and these show that it still ends the wait.
describe("the local deadline", () => {
  it("rejects with expired when the code's lifetime passes, even if the server never says so", async () => {
    // The server stays pending: its clock is the real one, which this test never moves.
    galaxy = await start({ interval: 100, deviceExpiresIn: 250 });
    const { server, net } = await connect();
    const w = startWaiting(await server.startDeviceLogin(), net);
    w.done.catch(() => undefined); // it rejects while the clock is still being moved

    await w.tick(100_000 + JUST_AFTER);
    expect(tokenRequests()).toBeGreaterThanOrEqual(1);
    await vi.advanceTimersByTimeAsync(200_000);
    expect((await authErrorOf(w.done)).kind).toBe("expired");
  });

  it("still ends at the deadline when the server keeps failing, so only the local timer can end it", async () => {
    galaxy = await start({ interval: 100, deviceExpiresIn: 250 });
    const { server, net } = await connect();
    galaxy.failTokenPolls(1000, { status: 502 });
    const w = startWaiting(await server.startDeviceLogin(), net);
    w.done.catch(() => undefined); // it rejects while the clock is still being moved

    await w.tick(100_000 + JUST_AFTER);
    expect(tokenRequests()).toBe(1);
    await vi.advanceTimersByTimeAsync(300_000);
    const e = await authErrorOf(w.done);
    expect(e.kind).toBe("expired");
    // An outage that outlasts the code must not read as nobody approving it.
    expect(e.message).toContain("HTTP 502");
  });

  // A request in flight at the deadline is ended with it, and not left for the library's own 30 s
  // timeout. Real timers: the code's lifetime is the shortest the client allows, 1 s.
  it("ends a poll that never answers when the code expires", async () => {
    galaxy = await start({ interval: 0.01, deviceExpiresIn: 1 });
    const server = await discover(url());
    galaxy.failTokenPolls(1000, { hang: true });
    const pending = await server.startDeviceLogin();
    const started = Date.now();
    expect((await authErrorOf(pending.waitForApproval())).kind).toBe("expired");
    expect(Date.now() - started).toBeLessThan(5000);
  });
});

// The server chooses these numbers, and `setTimeout` fires at once for more than 2^31 ms.
describe("numbers the server chooses", () => {
  it("polls an interval of years every 60 s at most", async () => {
    galaxy = await start({ interval: 3e6 });
    const { server, net } = await connect();
    const w = startWaiting(await server.startDeviceLogin(), net);
    w.done.catch(() => undefined);
    await w.tick(60_000 + JUST_AFTER);
    expect(tokenRequests()).toBe(1);
    galaxy.approve(galaxy.pendingCodes()[0]!);
    await vi.advanceTimersByTimeAsync(60_000 + JUST_AFTER);
    await expect(w.done).resolves.toBeDefined();
  });

  it("never polls sooner than a second, however small an interval is asked for", async () => {
    galaxy = await start({ interval: 0.001 });
    const net = tracked();
    const server = await discoverWith(url(), { fetch: net.fetch }); // the real floor
    const w = startWaiting(await server.startDeviceLogin(), net);
    w.done.catch(() => undefined);
    await w.tick(900);
    expect(tokenRequests()).toBe(0);
    await w.tick(200);
    expect(tokenRequests()).toBe(1);
    galaxy.approve(galaxy.pendingCodes()[0]!);
    await vi.advanceTimersByTimeAsync(1100);
    await expect(w.done).resolves.toBeDefined();
  });

  it("caps a Retry-After of years at 60 s", async () => {
    galaxy = await start({ interval: 1 });
    const { server, net } = await connect();
    galaxy.failTokenPolls(1, { status: 429, headers: { "retry-after": "99999999999" } });
    const w = startWaiting(await server.startDeviceLogin(), net);
    w.done.catch(() => undefined);
    await w.tick(1000 + JUST_AFTER);
    expect(tokenRequests()).toBe(1);
    await w.tick(58_000);
    expect(tokenRequests()).toBe(1);
    await w.tick(2000 + JUST_AFTER);
    expect(tokenRequests()).toBe(2);
    galaxy.approve(galaxy.pendingCodes()[0]!);
    await vi.advanceTimersByTimeAsync(1100);
    await expect(w.done).resolves.toBeDefined();
  });

  // RFC 9110 §10.2.3 allows an HTTP date. The clock `Date.now` reads is real here (only timers are
  // faked), so the date is 30 s from the real now.
  it("honours a Retry-After given as an HTTP date", async () => {
    galaxy = await start({ interval: 1 });
    const { server, net } = await connect();
    const at = new Date(Date.now() + 30_000).toUTCString();
    galaxy.failTokenPolls(1, { status: 429, headers: { "retry-after": at } });
    const w = startWaiting(await server.startDeviceLogin(), net);
    w.done.catch(() => undefined);
    await w.tick(1000 + JUST_AFTER);
    expect(tokenRequests()).toBe(1);
    await w.tick(25_000);
    expect(tokenRequests()).toBe(1);
    await w.tick(6000);
    expect(tokenRequests()).toBe(2);
    galaxy.approve(galaxy.pendingCodes()[0]!);
    await vi.advanceTimersByTimeAsync(1100);
    await expect(w.done).resolves.toBeDefined();
  });
});

// RFC 8628 §3.5 from the server's side: the fake answers `slow_down` to a poll that comes before
// the interval is up, so a client that ignored `interval` would be told so.
describe("a server that enforces the interval", () => {
  it("is never polled too fast", async () => {
    galaxy = await start({ interval: 0.1, enforceInterval: true });
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    const done = pending.waitForApproval();
    await vi.waitFor(() => {
      expect(tokenRequests()).toBeGreaterThanOrEqual(3);
    });
    galaxy.approve(pending.userCode);
    await done;
    expect(galaxy.tooFastPolls()).toBe(0);
  });
});

// A login waits as long as the code lives, so one dropped connection or 5xx must not end it. What
// does end it at once is a failure that retrying cannot fix: an OAuth error that is not
// `authorization_pending` or `slow_down`, a 4xx other than 429, or a certificate error.
describe("transient failures", () => {
  beforeEach(async () => {
    galaxy = await start({ interval: 0.01 });
  });

  it("keeps polling through connection resets, then succeeds", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    galaxy.failTokenPolls(2, { reset: true });
    galaxy.approve(pending.userCode, { scopes: ["read"] });
    await expect(pending.waitForApproval()).resolves.toMatchObject({ scopes: ["read"] });
    expect(tokenRequests()).toBeGreaterThan(2);
  });

  it.each([502, 503, 504])("keeps polling through an HTML %i", async (status) => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    galaxy.failTokenPolls(2, { status });
    galaxy.approve(pending.userCode);
    const grant = await pending.waitForApproval();
    expect(galaxy.issuedTokens()).toContain(grant.accessToken);
  });

  it("keeps polling through a 500 with an OAuth error body", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    galaxy.failTokenPolls(1, { status: 500, body: JSON.stringify({ error: "server_error" }) });
    galaxy.approve(pending.userCode);
    const grant = await pending.waitForApproval();
    expect(galaxy.issuedTokens()).toContain(grant.accessToken);
  });

  it("fails at once on an OAuth error that is not pending or slow_down", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    galaxy.setTokenOverride({ status: 400, body: JSON.stringify({ error: "server_error" }) });
    expect((await authErrorOf(pending.waitForApproval())).kind).toBe("protocol");
    expect(tokenRequests()).toBe(1);
  });

  it("fails at once on a 4xx that is not 429", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    galaxy.failTokenPolls(5, { status: 403 });
    expect((await authErrorOf(pending.waitForApproval())).kind).toBe("protocol");
    expect(tokenRequests()).toBe(1);
  });

  it("fails at once on a certificate error, naming how to fix it", async () => {
    let attempts = 0;
    const tlsFetch: typeof fetch = (input, init) => {
      if (!urlOf(input).endsWith("/oauth/token")) return fetch(input, init);
      attempts += 1;
      return Promise.reject(
        Object.assign(new TypeError("fetch failed"), {
          cause: Object.assign(new Error("self-signed certificate"), {
            code: "DEPTH_ZERO_SELF_SIGNED_CERT",
          }),
        }),
      );
    };
    const server = await discover(url(), { fetch: tlsFetch });
    const pending = await server.startDeviceLogin();
    const e = await authErrorOf(pending.waitForApproval());
    expect(e.kind).toBe("network");
    expect(e.message).toContain("NODE_EXTRA_CA_CERTS");
    expect(attempts).toBe(1);
  });

  // A TypeError with a code is the library or Node refusing what we asked, not the network: retrying
  // would only repeat it until the code expired.
  it("fails at once on a coded TypeError, which is not the network", async () => {
    let attempts = 0;
    const broken: typeof fetch = (input, init) => {
      if (!urlOf(input).endsWith("/oauth/token")) return fetch(input, init);
      attempts += 1;
      return Promise.reject(
        Object.assign(new TypeError("bad argument"), { code: "ERR_INVALID_ARG_TYPE" }),
      );
    };
    const server = await discover(url(), { fetch: broken });
    const pending = await server.startDeviceLogin();
    await authErrorOf(pending.waitForApproval());
    expect(attempts).toBe(1);
  });

  it("waits out a 429's Retry-After before polling again", async () => {
    galaxy = await start({ interval: 1 });
    const { server, net } = await connect();
    galaxy.failTokenPolls(1, { status: 429, headers: { "retry-after": "20" } });
    const w = startWaiting(await server.startDeviceLogin(), net);
    w.done.catch(() => undefined);

    await w.tick(1000 + JUST_AFTER); // first poll: 429
    expect(tokenRequests()).toBe(1);
    await w.tick(19_000); // 20 s have not passed since the 429
    expect(tokenRequests()).toBe(1);
    await w.tick(2000 + JUST_AFTER);
    expect(tokenRequests()).toBe(2);

    galaxy.approve(galaxy.pendingCodes()[0]!);
    await vi.advanceTimersByTimeAsync(1000 + JUST_AFTER);
    await expect(w.done).resolves.toBeDefined();
  });
});

// A cancel must stop the polling at once and leave nothing on the server. The poll loop is the
// client's own because openid-client's waits ignore the signal for up to 5 s and its timers keep
// the process alive after a Ctrl-C (oauth.ts).
describe("cancelling", () => {
  beforeEach(async () => {
    galaxy = await start();
  });

  it("aborts at once when the signal is already aborted, without polling", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    const controller = new AbortController();
    controller.abort();
    await expect(pending.waitForApproval(controller.signal)).rejects.toMatchObject({
      name: "AbortError",
    });
    expect(tokenRequests()).toBe(0);
  });

  it("aborts polling with the signal's reason and polls no more", async () => {
    galaxy = await start({ interval: 1 });
    const { server, net } = await connect();
    const controller = new AbortController();
    const w = startWaiting(await server.startDeviceLogin(), net, controller.signal);
    w.done.catch(() => undefined);

    await w.tick(1000 + JUST_AFTER);
    expect(tokenRequests()).toBe(1);
    controller.abort(new Error("user pressed ctrl-c"));
    await expect(w.done).rejects.toThrow("user pressed ctrl-c");
    await vi.advanceTimersByTimeAsync(10_000);
    expect(tokenRequests()).toBe(1);
  });

  it("rejects with the reason when the signal aborts during discovery", async () => {
    const controller = new AbortController();
    const slowFetch: typeof fetch = (_input, init) =>
      new Promise((_resolve, reject) => {
        const signal = init?.signal;
        if (!signal) return;
        signal.addEventListener(
          "abort",
          () => {
            reject(abortReason(signal));
          },
          { once: true },
        );
      });
    const p = discover(url(), { fetch: slowFetch, signal: controller.signal });
    controller.abort(new Error("ctrl-c"));
    await expect(p).rejects.toThrow("ctrl-c");
  });

  it("creates no device code when the signal was aborted before startDeviceLogin", async () => {
    const controller = new AbortController();
    const server = await discover(url());
    controller.abort(new Error("ctrl-c"));
    await expect(server.startDeviceLogin(undefined, controller.signal)).rejects.toThrow("ctrl-c");
    expect(galaxy.requestsTo("/oauth/device_authorization")).toHaveLength(0);
  });

  // `openid-client` builds its own timeout for each request, so the signal has to be handed to the
  // `fetch` it calls. A signal given only to `waitForApproval` (discovery got none) must end a
  // request the server never answers, not wait for the library's 30 s.
  it("ends a poll the server never answers when only waitForApproval was given a signal", async () => {
    const server = await discover(url());
    galaxy.failTokenPolls(1, { hang: true });
    const pending = await server.startDeviceLogin();
    const controller = new AbortController();
    const done = pending.waitForApproval(controller.signal);
    done.catch(() => undefined);
    await vi.waitFor(() => {
      expect(tokenRequests()).toBe(1);
    });
    controller.abort(new Error("ctrl-c"));
    await expect(done).rejects.toThrow("ctrl-c");
  }, 5000);

  it("ends the device authorization request when only startDeviceLogin was given a signal", async () => {
    let seen: AbortSignal | undefined;
    const hanging: typeof fetch = (input, init) => {
      if (!urlOf(input).endsWith("/oauth/device_authorization")) return fetch(input, init);
      seen = init?.signal ?? undefined;
      return new Promise((_resolve, reject) => {
        seen?.addEventListener(
          "abort",
          () => {
            reject(abortReason(seen!));
          },
          { once: true },
        );
      });
    };
    const server = await discover(url(), { fetch: hanging });
    const controller = new AbortController();
    const p = server.startDeviceLogin(undefined, controller.signal);
    p.catch(() => undefined);
    await vi.waitFor(() => {
      expect(seen).toBeDefined();
    });
    controller.abort(new Error("ctrl-c"));
    await expect(p).rejects.toThrow("ctrl-c");
  }, 5000);

  // The signal that cancelled a login must not reach a revocation, or a token minted just before
  // the Ctrl-C could not be taken back (login.ts). The signal given to discovery is the login's.
  it("does not let the discovery signal reach a later revocation", async () => {
    const controller = new AbortController();
    const server = await discover(url(), { signal: controller.signal });
    const token = galaxy.issueToken();
    controller.abort();
    await server.revoke(token);
    expect(galaxy.isTokenValid(token)).toBe(false);
  });

  it("ends a revocation the server never answers when revoke's own signal aborts", async () => {
    let asked = false;
    const hanging: typeof fetch = (input, init) => {
      if (!urlOf(input).endsWith("/oauth/revoke")) return fetch(input, init);
      asked = true;
      return new Promise((_resolve, reject) => {
        init?.signal?.addEventListener(
          "abort",
          () => {
            reject(abortReason(init.signal!));
          },
          { once: true },
        );
      });
    };
    const server = await discover(url(), { fetch: hanging });
    const controller = new AbortController();
    const p = server.revoke("fake-access-token-1", controller.signal);
    p.catch(() => undefined);
    await vi.waitFor(() => {
      expect(asked).toBe(true);
    });
    controller.abort(new Error("gave up"));
    await expect(p).rejects.toThrow();
  });
});
