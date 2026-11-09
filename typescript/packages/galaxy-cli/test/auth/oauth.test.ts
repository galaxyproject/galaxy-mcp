/**
 * The OAuth client against the fake Galaxy over a real socket: what it asks for, what it refuses
 * and how it classifies what comes back. How it waits, and how it is cancelled, is in
 * oauth-polling.test.ts.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { discover as discoverWith, type DiscoverOptions } from "../../src/auth/oauth";
import { CLIENT_ID, type AuthServer, type BaseUrl } from "../../src/auth/types";
import { startFakeGalaxy, useFakeGalaxies, type FakeGalaxy } from "../support/fake-galaxy";
import { authErrorOf } from "./harness";

/** Discovery with no floor on the poll interval: the fake asks for 10 ms, and the real floor is 1 s. */
const discover = (baseUrl: BaseUrl, options: DiscoverOptions = {}) =>
  discoverWith(baseUrl, { minIntervalMs: 0, ...options });

const start = useFakeGalaxies();
let galaxy: FakeGalaxy;
const url = () => galaxy.baseUrl as BaseUrl;

describe("discover", () => {
  beforeEach(async () => {
    galaxy = await start();
  });

  it("reads the metadata and reports what the server offers", async () => {
    const server = await discover(url());
    expect(server.issuer).toBe(galaxy.baseUrl);
    expect(server.supportsDeviceFlow).toBe(true);
    expect(server.supportsRevocation).toBe(true);
    expect(galaxy.requests[0]?.url).toBe("/.well-known/oauth-authorization-server");
  });

  it("reports no device flow when the metadata has no device endpoint, and refuses to start one", async () => {
    galaxy.setDeviceFlow(false);
    const server = await discover(url());
    expect(server.supportsDeviceFlow).toBe(false);
    expect((await authErrorOf(server.startDeviceLogin())).kind).toBe("unsupported");
    expect(galaxy.requestsTo("/oauth/device_authorization")).toHaveLength(0);
  });

  it("reports no revocation when the metadata has no revocation endpoint", async () => {
    galaxy.setRevocationEndpoint(false);
    const server = await discover(url());
    expect(server.supportsRevocation).toBe(false);
    expect((await authErrorOf(server.revoke("fake-access-token-1"))).kind).toBe("unsupported");
  });

  // `unsupported` is what a caller turns into the hint to paste a key instead. Broken metadata
  // (`protocol`) and metadata for another issuer (`insecure`, RFC 8414 §3.3) are not a Galaxy that
  // merely lacks the feature, so they are other kinds and get no such hint.
  it.each([
    ["404 (what Galaxy 26.1 sends)", "not-found", "unsupported"],
    ["a page that is not JSON", "not-json", "unsupported"],
    ["malformed JSON", "malformed", "protocol"],
    ["another issuer", "wrong-issuer", "insecure"],
  ] as const)("classifies metadata that is %s as %s", async (_what, mode, kind) => {
    galaxy.setMetadataMode(mode);
    expect((await authErrorOf(discover(url()))).kind).toBe(kind);
  });

  // Only "there is no such document" (a 404, or a page that is not JSON) is a Galaxy without device
  // login. A proxy that is briefly down is not, and a redirect is somebody else's answer.
  it("is a network error, not 'unsupported', when the metadata answers 503", async () => {
    galaxy.setMetadataMode("unavailable");
    const e = await authErrorOf(discover(url()));
    expect(e.kind).toBe("network");
    expect(e.message).toContain("503");
  });

  // `setRedirect` answers with any status, which is all these two need.
  it.each([
    ["429", "network"],
    ["403", "protocol"],
  ] as const)("classifies a %s from the metadata as %s", async (status, kind) => {
    galaxy.setRedirect("/.well-known/oauth-authorization-server", "/x", Number(status));
    const e = await authErrorOf(discover(url()));
    expect(e.kind).toBe(kind);
    expect(e.message).toContain(status);
  });

  it("is a protocol error naming the host, not 'unsupported', when the metadata redirects", async () => {
    galaxy.setRedirect(
      "/.well-known/oauth-authorization-server",
      "https://login.sso.example/start?next=%0A",
    );
    const e = await authErrorOf(discover(url()));
    expect(e.kind).toBe("protocol");
    expect(e.message).toContain("login.sso.example");
    // Nothing was followed.
    expect(galaxy.requests).toHaveLength(1);
  });

  it("is a protocol error when a device endpoint comes with no token endpoint", async () => {
    galaxy.setMetadataOverrides({ token_endpoint: undefined });
    const e = await authErrorOf(discover(url()));
    expect(e.kind).toBe("protocol");
    expect(e.message).toContain("token_endpoint");
    expect(galaxy.requestsTo("/oauth/device_authorization")).toHaveLength(0);
  });

  it("is a network error, naming the CA setting and not the proxy, when nothing is listening", async () => {
    const dead = await startFakeGalaxy();
    const deadUrl = dead.baseUrl as BaseUrl;
    await dead.close();
    const e = await authErrorOf(discover(deadUrl));
    expect(e.kind).toBe("network");
    expect(e.message).not.toContain("NODE_USE_ENV_PROXY");
    expect(e.message).toContain("NODE_EXTRA_CA_CERTS");
  });
});

describe("a Galaxy under a path prefix", () => {
  it("finds the metadata by RFC 8414 path insertion", async () => {
    galaxy = await start({ pathPrefix: "/galaxy" });
    const server = await discover(url());
    expect(server.issuer).toBe(galaxy.baseUrl);
    expect(galaxy.requests[0]?.url).toBe("/.well-known/oauth-authorization-server/galaxy");

    const pending = await server.startDeviceLogin();
    expect(pending.verificationUri.startsWith(`${galaxy.baseUrl}/`)).toBe(true);
    expect(galaxy.requestsTo("/galaxy/oauth/device_authorization")).toHaveLength(1);
  });
});

// Polling sends the device code to the token endpoint and logout sends the token to the revocation
// endpoint, so metadata that points any of the three at another origin is refused before anything
// is sent (oauth.ts).
describe("endpoints must be on the issuer's origin", () => {
  beforeEach(async () => {
    galaxy = await start();
  });

  it.each(["device_authorization_endpoint", "token_endpoint", "revocation_endpoint"])(
    "refuses a %s on another origin",
    async (field) => {
      galaxy.setMetadataOverrides({ [field]: "http://127.0.0.1:1/elsewhere" });
      const e = await authErrorOf(discover(url()));
      expect(e.kind).toBe("insecure");
      expect(e.message).toContain(field);
    },
  );

  it("refuses an endpoint that is not a URL", async () => {
    galaxy.setMetadataOverrides({ token_endpoint: "not a url" });
    expect((await authErrorOf(discover(url()))).kind).toBe("insecure");
  });

  it("accepts endpoints on the same origin under another path", async () => {
    galaxy.setMetadataOverrides({ revocation_endpoint: `${galaxy.baseUrl}/somewhere/else` });
    await expect(discover(url())).resolves.toMatchObject({ supportsRevocation: true });
  });
});

describe("device login", () => {
  beforeEach(async () => {
    galaxy = await start();
  });

  it("runs the happy path: code out, approval in, token with granted scopes", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin(["read", "write"]);
    expect(pending.userCode).toBe(galaxy.pendingCodes()[0]);
    expect(pending.verificationUri).toBe(`${galaxy.baseUrl}/oauth/device`);
    expect(pending.expiresAt.getTime()).toBeGreaterThan(Date.now());

    galaxy.approve(pending.userCode, { scopes: ["read"], expiresIn: 1800 });
    const grant = await pending.waitForApproval();
    expect(grant).toEqual({
      accessToken: galaxy.issuedTokens()[0],
      expiresIn: 1800,
      scopes: ["read"],
    });
  });

  // The client is public: it holds no secret and says so, so there is nothing to leak to a server
  // that asks for one.
  it("sends the public client id and scope, and no secret or Authorization header", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin(["read", "write"]);
    galaxy.approve(pending.userCode);
    await pending.waitForApproval();

    const device = galaxy.requestsTo("/oauth/device_authorization")[0]!;
    expect(device.form).toEqual({ client_id: CLIENT_ID, scope: "read write" });
    const token = galaxy.requestsTo("/oauth/token")[0]!;
    expect(token.form.client_id).toBe(CLIENT_ID);
    expect(token.form.grant_type).toBe("urn:ietf:params:oauth:grant-type:device_code");
    for (const r of galaxy.requests) {
      expect(r.headers.authorization).toBeUndefined();
      expect(r.form.client_secret).toBeUndefined();
      expect(r.form.client_assertion).toBeUndefined();
    }
  });

  // No `scope` parameter at all, not an empty one, is how the client asks for the server's default.
  it("sends scope only when scopes are given", async () => {
    const server = await discover(url());
    await server.startDeviceLogin();
    await server.startDeviceLogin([]);
    for (const r of galaxy.requestsTo("/oauth/device_authorization"))
      expect(r.form).toEqual({ client_id: CLIENT_ID });
  });

  // The fake knows one client id and answers anything else `invalid_client`, on all three
  // endpoints, so a request that dropped it would fail here.
  it("uses the client id from the options on every request", async () => {
    galaxy = await start({ clientId: "other-client" });
    const server = await discover(url(), { clientId: "other-client" });
    const pending = await server.startDeviceLogin();
    galaxy.approve(pending.userCode);
    const grant = await pending.waitForApproval();
    await server.revoke(grant.accessToken);
    for (const suffix of ["/oauth/device_authorization", "/oauth/token", "/oauth/revoke"])
      expect(galaxy.requestsTo(suffix)[0]!.form.client_id).toBe("other-client");
    expect(galaxy.isTokenValid(grant.accessToken)).toBe(false);
  });

  it("gets invalid_client from a Galaxy that does not know the client id", async () => {
    const server = await discover(url(), { clientId: "unregistered" });
    const e = await authErrorOf(server.startDeviceLogin());
    expect(e.kind).toBe("protocol");
    expect(e.message).toContain("invalid_client");
  });

  // RFC 6749 §5.1: a server omits `scope` when it granted what was requested, so absence is not "none".
  it("falls back to the requested scopes when the response has no scope, and has no expiry without expires_in", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin(["read"]);
    galaxy.approve(pending.userCode, { scopes: undefined, expiresIn: null });
    const grant = await pending.waitForApproval();
    expect(grant.scopes).toEqual(["read"]);
    expect(grant.expiresIn).toBeUndefined();
  });

  it("has no scopes when none were requested or returned", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    galaxy.approve(pending.userCode, { scopes: undefined });
    expect((await pending.waitForApproval()).scopes).toEqual([]);
  });

  it("rejects with denied when the user denies", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    galaxy.deny(pending.userCode);
    expect((await authErrorOf(pending.waitForApproval())).kind).toBe("denied");
  });

  // The server chooses `verification_uri` and a person is sent there, so the client checks it
  // (verification-uri.ts) and a caller never sees one that is off the Galaxy.
  describe("the approval page", () => {
    const deviceResponse = (verification_uri: string) =>
      JSON.stringify({
        device_code: "fake-device-code-x",
        user_code: "FAKE-9999",
        verification_uri,
        expires_in: 600,
        interval: 5,
      });

    it.each([
      ["another host", "https://evil.example/oauth/device"],
      ["another port", "http://127.0.0.1:1/oauth/device"],
      ["a non-http scheme", "javascript:alert(1)"],
      ["not a URL", "nonsense"],
    ])("is refused as insecure when it is %s", async (_what, uri) => {
      galaxy.setDeviceAuthorizationOverride({ status: 200, body: deviceResponse(uri) });
      const server = await discover(url());
      const e = await authErrorOf(server.startDeviceLogin());
      expect(e.kind).toBe("insecure");
    });

    it("is returned as the URL parser normalises it, without the tab and newline it strips", async () => {
      galaxy.setDeviceAuthorizationOverride({
        status: 200,
        body: deviceResponse(`${galaxy.baseUrl}/oauth/de\tvi\nce`),
      });
      const server = await discover(url());
      const pending = await server.startDeviceLogin();
      expect(pending.verificationUri).toBe(`${galaxy.baseUrl}/oauth/device`);
    });
  });

  // `setTimeout` fires at once for more than 2^31 ms, so a server's huge number must not become a
  // hot loop or an instant expiry.
  it("clamps an expires_in of years to an hour", async () => {
    galaxy.setDeviceAuthorizationOverride({
      status: 200,
      body: JSON.stringify({
        device_code: "fake-device-code-x",
        user_code: "FAKE-9999",
        verification_uri: `${galaxy.baseUrl}/oauth/device`,
        expires_in: 3e9,
      }),
    });
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    const seconds = (pending.expiresAt.getTime() - Date.now()) / 1000;
    expect(seconds).toBeGreaterThan(3500);
    expect(seconds).toBeLessThanOrEqual(3600);
  });

  it("rejects with expired when the server says the code expired", async () => {
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    galaxy.expire(pending.userCode);
    expect((await authErrorOf(pending.waitForApproval())).kind).toBe("expired");
  });

  it.each([
    ["malformed JSON", "{nope"],
    ["no user code", JSON.stringify({ device_code: "x", verification_uri: "y", expires_in: 5 })],
  ])("rejects with protocol when the device authorization response has %s", async (_what, body) => {
    galaxy.setDeviceAuthorizationOverride({ status: 200, body });
    const server = await discover(url());
    expect((await authErrorOf(server.startDeviceLogin())).kind).toBe("protocol");
  });
});

// A token goes into an HTTP header and, for `auth env`, into a shell (token-shape.ts), so one with
// whitespace, control characters or non-ASCII is a broken response or a hostile one. It is refused
// without being quoted back, because it is a secret and not one that can be used.
describe("the token Galaxy issues", () => {
  beforeEach(async () => {
    galaxy = await start();
  });

  async function grantWith(accessToken: string) {
    const server = await discover(url());
    const pending = await server.startDeviceLogin();
    galaxy.approve(pending.userCode, { accessToken });
    return pending.waitForApproval();
  }

  it("is used when it is printable ASCII without whitespace", async () => {
    await expect(grantWith("fake-ok_token.1")).resolves.toMatchObject({
      accessToken: "fake-ok_token.1",
    });
  });

  const unusable: [string, string][] = [
    ["whitespace", "abc def"],
    ["a curly quote", "abc‘def"],
    ["a control character", "tok\u0000en"],
    ["non-ASCII letters", "tökén"],
  ];

  it.each([...unusable, ["nothing at all", ""]])(
    "is refused as a protocol error when it has %s",
    async (_what, bad) => {
      const e = await authErrorOf(grantWith(bad));
      expect(e.kind).toBe("protocol");
    },
  );

  // The empty token is left out: every message contains "", so it cannot be "not echoed".
  it.each(unusable)("is not echoed in the error when it has %s", async (_what, bad) => {
    const e = await authErrorOf(grantWith(bad));
    expect(e.message).not.toContain(bad);
  });
});

describe("revocation", () => {
  beforeEach(async () => {
    galaxy = await start();
  });

  // RFC 7009 §2.1: a form post carrying `token` and `token_type_hint`; a public client sends no credentials.
  it("sends the token as a form post to the revocation endpoint, with no Authorization header", async () => {
    const server = await discover(url());
    const token = galaxy.issueToken();
    await server.revoke(token);
    const req = galaxy.requestsTo("/oauth/revoke")[0]!;
    expect(req.form).toEqual({ token, token_type_hint: "access_token", client_id: CLIENT_ID });
    expect(req.headers.authorization).toBeUndefined();
  });

  it("treats a token Galaxy does not know as revoked (RFC 7009 answers 200)", async () => {
    const server = await discover(url());
    await expect(server.revoke("fake-unknown-token")).resolves.toBeUndefined();
  });
});

// What the OAuth client refuses or hides: redirects with credentials, plain http, hostile server text.

describe("no request reaches another origin through a redirect", () => {
  // A client that followed a redirect would carry the device code, the token or the key to an origin
  // the person never chose. `redirect()` makes the Galaxy answer the named path with a redirect to
  // another origin, which stands for anything a hostile or misconfigured Galaxy could point at.
  type Step = { galaxy: FakeGalaxy; server: AuthServer; redirect: () => void };
  const rows: [what: string, path: string, act: (s: Step) => Promise<unknown>][] = [
    [
      "the metadata",
      "/.well-known/oauth-authorization-server",
      ({ galaxy, redirect }) => (redirect(), discover(galaxy.baseUrl as BaseUrl)),
    ],
    [
      "the device authorization endpoint",
      "/oauth/device_authorization",
      ({ server, redirect }) => (redirect(), server.startDeviceLogin()),
    ],
    [
      "the token endpoint",
      "/oauth/token",
      async ({ server, redirect }) => {
        const pending = await server.startDeviceLogin();
        redirect();
        return pending.waitForApproval();
      },
    ],
    [
      "the revocation endpoint",
      "/oauth/revoke",
      ({ server, redirect }) => (redirect(), server.revoke("fake-access-token-1")),
    ],
  ];

  it.each(rows)("a redirect from %s is refused", async (_what, path, act) => {
    const galaxy = await start();
    const other = await start();
    const server = await discover(galaxy.baseUrl as BaseUrl);
    await authErrorOf(
      act({
        galaxy,
        server,
        redirect: () => {
          galaxy.setRedirect(path, `${other.baseUrl}${path}`);
        },
      }),
    );
    // No device code, token or key was sent to the other origin.
    expect(other.requests).toEqual([]);
  });
});

describe("plain http", () => {
  // A token would cross the network unencrypted. The spy shows that the refusal comes before any request.
  it("is refused to a non-loopback host without making a request", async () => {
    const fetchSpy = vi.fn<typeof fetch>();
    const e = await authErrorOf(
      discover("http://galaxy.example.org" as BaseUrl, { fetch: fetchSpy }),
    );
    expect(e.kind).toBe("insecure");
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

describe("network and TLS errors name the fixes", () => {
  // The fix named for a private CA is to trust it. Turning verification off is never suggested: it
  // is the check that protects the token this connection is about to carry.
  it("a certificate failure names NODE_EXTRA_CA_CERTS and never suggests turning verification off", async () => {
    const cause = Object.assign(new Error("self-signed certificate in certificate chain"), {
      code: "SELF_SIGNED_CERT_IN_CHAIN",
    });
    const failing = (async () => {
      throw new TypeError("fetch failed", { cause });
    }) as unknown as typeof fetch;
    const e = await authErrorOf(
      discover("https://galaxy.example.org" as BaseUrl, { fetch: failing }),
    );
    expect(e.kind).toBe("network");
    expect(e.message).toContain("self-signed certificate");
    expect(e.message).toContain("NODE_EXTRA_CA_CERTS");
    expect(e.message).not.toContain("NODE_USE_ENV_PROXY");
    expect(e.message).not.toMatch(/NODE_TLS_REJECT_UNAUTHORIZED|insecure|--no-verify|disable/i);
  });
});

describe("server text in errors is sanitised", () => {
  // Server text reaches the person's terminal, and a terminal acts on escape sequences: this
  // response tries to retitle the window and clear the screen from inside an error message.
  it("strips terminal control characters from an OAuth error and its description", async () => {
    const galaxy = await start();
    galaxy.setDeviceAuthorizationOverride({
      status: 400,
      body: JSON.stringify({
        error: "bad\u001b]0;pwned\u0007thing",
        error_description: "line\r\nbreak \u001b[2J",
      }),
    });
    const server = await discover(galaxy.baseUrl as BaseUrl);
    const e = await authErrorOf(server.startDeviceLogin());
    expect(e.message).toContain("bad]0;pwnedthing");
    expect(e.message).not.toMatch(/[\u0000-\u001f\u007f-\u009f]/);
  });
});
