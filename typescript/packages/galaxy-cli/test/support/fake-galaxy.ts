/**
 * A fake Galaxy for testing `gxy auth`: an RFC 8414 authorization server (device flow,
 * token, revocation) plus `GET /api/version` and `GET /api/users/current`, on a `node:http`
 * server bound to 127.0.0.1 with an OS-assigned port.
 *
 * What is real and what is a guess. Confirmed against a real Galaxy 26.1: an unknown key at
 * `/api/users/current` gets 200 with an anonymous record that has no `username`;
 * `/.well-known/oauth-authorization-server` is a 404 with an HTML body; `/api/version` is
 * JSON with `version_major`. Everything else is an assumption about what Galaxy will do, not
 * an observation: the whole device-flow contract (metadata, `/oauth/device_authorization`,
 * `/oauth/token`, `/oauth/revoke`; the contract proposed to Galaxy upstream), the exact
 * fields of the anonymous record and of a user record, the 403 shape of
 * `setInvalidKeyResponse("forbidden")`, and the redirect and error faults. Tests should
 * assert the behaviour of our code against these, not the literals.
 *
 * - metadata at `/.well-known/oauth-authorization-server<prefix>` (RFC 8414 path
 *   insertion); the issuer is the base URL
 * - public client `galaxy-cli` (no secret; `token_endpoint_auth_methods_supported: ["none"]`)
 * - RFC 8628 device authorization, without `verification_uri_complete`
 * - a token endpoint answering `authorization_pending`, `slow_down`, `access_denied`,
 *   `expired_token`, or the token
 * - RFC 7009 revocation
 * - `/api/users/current`, which knows a user only by a token this server issued and has not
 *   revoked or expired
 *
 * It is a real HTTP server, not a stubbed `fetch`, for two reasons. The built binary in
 * e2e.int.test.ts is a separate process and cannot see anything mocked in this one. And the
 * redirects, connection resets and timing that the client must survive are only real when
 * there is a socket.
 *
 * Everything a test needs to steer is a method on the returned object, and every request is
 * recorded in `requests`. Tokens are obviously fake (`fake-access-token-N`).
 *
 *   const galaxy = await startFakeGalaxy();
 *   const token = galaxy.issueToken({ admin: true });   // no device flow needed
 *   // or, for the flow itself:
 *   const code = galaxy.pendingCodes()[0];               // after startDeviceLogin
 *   galaxy.approve(code, { scopes: ["read"], expiresIn: 3600 });
 *   await galaxy.close();
 */
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach } from "vitest";

export interface FakeGalaxyOptions {
  /** Serve Galaxy under this path prefix, e.g. `/galaxy`. Default none. */
  pathPrefix?: string;
  /** Seconds the server asks clients to wait between polls. Fractions are fine for fast tests. Default 0.01. */
  interval?: number;
  /** Seconds a user code lives. Default 600. */
  deviceExpiresIn?: number;
  /** Whether the metadata lists a `revocation_endpoint`. Default true. */
  revocationEndpoint?: boolean;
  /** The one `client_id` the server knows. Default `galaxy-cli`; any other is `invalid_client`. */
  clientId?: string;
  /**
   * Answer `slow_down` to a poll that comes sooner than `interval` after the last one (RFC 8628
   * §3.5), as a real server may. Off by default because tests that fake the clock send their polls
   * in quick succession in real time, which would trip it.
   */
  enforceInterval?: boolean;
}

export interface Approval {
  /** Scopes granted; `undefined` makes the token response omit `scope`. Default `[]`. */
  scopes?: string[];
  /** Token lifetime in seconds; `null` omits `expires_in`. Default 3600. */
  expiresIn?: number | null;
  username?: string;
  email?: string;
  /** Make `/api/users/current` report `is_admin: true` for this token. */
  admin?: boolean;
  /** Issue this literal string as the access token (a canary a test can search for). */
  accessToken?: string;
}

/**
 * One request as the server received it. The body and its parsed form are kept so a test can
 * show what was sent (the client id, no `client_secret`) and where a token went (in a URL, in a
 * header, in a body), not only that a request was made.
 */
export interface RecordedRequest {
  method: string;
  /** Path and query as sent, including any prefix. */
  url: string;
  /** Lower-cased header names. */
  headers: Record<string, string | string[] | undefined>;
  /** Raw body. */
  body: string;
  /** The body parsed as `application/x-www-form-urlencoded` (empty when there is none). */
  form: Record<string, string>;
}

/**
 * How `/api/version` answers, for the check that runs before a token is sent anywhere: the
 * version (`ok`), a 302 to another origin (`redirect`, as a wrong address such as http for
 * https or a missing path prefix often answers), or a 200 HTML page (`html`, as a Galaxy behind
 * a `remote_user` proxy answers). The last two are the answers a token must never follow.
 */
export type VersionMode = "ok" | "redirect" | "html";

/**
 * How `/.well-known/oauth-authorization-server...` answers. `not-found` is the 404 HTML page
 * Galaxy 26.1 sends today and `not-json` a 200 HTML page: no metadata at all, so the client
 * says device login is unsupported. `malformed` is a 200 with broken JSON, a protocol fault.
 * `wrong-issuer` is metadata whose `issuer` is another server's, which RFC 8414 §3.3 says
 * must not be used, or a hostile Galaxy could point a client at someone else's endpoints.
 * `unavailable` is a 503 HTML page, as a proxy answers while Galaxy restarts: not a Galaxy
 * without device login, so it must not read as one.
 */
export type MetadataMode =
  "ok" | "not-found" | "not-json" | "malformed" | "wrong-issuer" | "unavailable";

export interface FakeGalaxy {
  /** `http://127.0.0.1:<port><pathPrefix>`: the Galaxy base URL and the issuer. */
  readonly baseUrl: string;
  readonly requests: RecordedRequest[];
  /** How many polls arrived sooner than `interval` after the one before (with `enforceInterval`). */
  tooFastPolls(): number;
  /** Requests whose URL ends with `suffix` (a path such as `/oauth/token`). */
  requestsTo(suffix: string): RecordedRequest[];
  /** User codes issued and not yet approved, denied or expired, oldest first. */
  pendingCodes(): string[];
  /**
   * Every device code (the secret one the CLI polls with) issued so far, in order. It is exposed
   * so a test can search the CLI's output for it: whoever holds it can collect the token once
   * the user code is approved, so it must never be printed.
   */
  deviceCodes(): string[];
  approve(userCode: string, approval?: Approval): void;
  deny(userCode: string): void;
  /** The code stops working: the token endpoint answers `expired_token`. */
  expire(userCode: string): void;
  /**
   * The next `n` token polls answer `slow_down` (before anything else), which RFC 8628 §3.5 says
   * adds 5 s to the interval.
   */
  slowDown(n: number): void;
  setDeviceFlow(on: boolean): void;
  setRevocationEndpoint(on: boolean): void;
  setMetadataMode(mode: MetadataMode): void;
  /** What `/api/version` answers: the version (default), a 302 to another origin, or an HTML page. */
  setVersionMode(mode: VersionMode): void;
  /** What `/api/users/current` says for an unknown key: 200 with the anonymous shape (default, observed), or 403 (a guess). */
  setInvalidKeyResponse(kind: "anonymous" | "forbidden"): void;
  /** Make the device authorization endpoint answer this response instead (e.g. malformed JSON). */
  setDeviceAuthorizationOverride(
    r: { status: number; body: string; contentType?: string } | null,
  ): void;
  /**
   * Make the token endpoint answer this response while the code is still pending. A token fault
   * or a `slow_down` still comes first.
   */
  setTokenOverride(r: { status: number; body: string; contentType?: string } | null): void;
  /**
   * Answer every request whose path ends with `pathSuffix` with a redirect to `location`
   * (default 302), before anything else; `null` removes it. For proving a client does not
   * follow a redirect with credentials attached.
   */
  setRedirect(pathSuffix: string, location: string | null, status?: number): void;
  /**
   * The next `n` token polls fail before anything else: `{ reset: true }` destroys the
   * socket, `{ hang: true }` never answers (until the client gives up), otherwise the response is
   * `status` with the given headers (a 502, a 429...).
   */
  failTokenPolls(
    n: number,
    fault:
      | { reset: true }
      | { hang: true }
      | { status: number; headers?: Record<string, string>; body?: string },
  ): void;
  /** Merge these fields over the discovery metadata (e.g. an endpoint on another origin). */
  setMetadataOverrides(fields: Record<string, unknown> | null): void;
  /** Make `/api/users/current` answer this status with an empty JSON body; `null` restores it. */
  setUsersCurrentStatus(status: number | null): void;
  /** Make `/api/users/current` never answer, so a test can cancel a login while its whoami is out. */
  hangUsersCurrent(on: boolean): void;
  /** Issue a token as a finished device login would, without running the flow. */
  issueToken(options?: Approval): string;
  /** Tokens issued so far, in order, revoked or not. */
  issuedTokens(): string[];
  /** Whether this token would be accepted by `/api/users/current` right now. */
  isTokenValid(token: string): boolean;
  close(): Promise<void>;
}

/** One device login the fake has issued; a code is good for one token, and then `invalid_grant`. */
interface DeviceEntry {
  userCode: string;
  deviceCode: string;
  expiresAtMs: number;
  state: "pending" | "approved" | "denied" | "expired";
  approval?: Approval;
  consumed: boolean;
}

interface TokenEntry {
  token: string;
  username: string;
  email: string;
  admin: boolean;
  expiresAtMs: number | null;
  revoked: boolean;
}

const DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code";

export async function startFakeGalaxy(options: FakeGalaxyOptions = {}): Promise<FakeGalaxy> {
  const prefix = (options.pathPrefix ?? "").replace(/\/+$/, "");
  const interval = options.interval ?? 0.01;
  const knownClientId = options.clientId ?? "galaxy-cli";
  let lastPollMs: number | undefined;
  let tooFast = 0;
  const deviceExpiresIn = options.deviceExpiresIn ?? 600;
  let deviceFlow = true;
  let revocation = options.revocationEndpoint ?? true;
  let metadataMode: MetadataMode = "ok";
  let versionMode: VersionMode = "ok";
  let invalidKey: "anonymous" | "forbidden" = "anonymous";
  let deviceOverride: { status: number; body: string; contentType?: string } | null = null;
  let tokenOverride: { status: number; body: string; contentType?: string } | null = null;
  let slowDowns = 0;
  const redirects = new Map<string, { location: string; status: number }>();

  const requests: RecordedRequest[] = [];
  const devices = new Map<string, DeviceEntry>(); // by user code
  const tokens = new Map<string, TokenEntry>();
  let counter = 0;
  // Set once the server is listening and has a port: the metadata and the codes need it.
  let baseUrl = "";

  const json = (
    res: ServerResponse,
    status: number,
    body: unknown,
    extra: Record<string, string> = {},
  ) => {
    res.writeHead(status, {
      "content-type": "application/json",
      "cache-control": "no-store",
      ...extra,
    });
    res.end(JSON.stringify(body));
  };
  const raw = (res: ServerResponse, r: { status: number; body: string; contentType?: string }) => {
    res.writeHead(r.status, { "content-type": r.contentType ?? "application/json" });
    res.end(r.body);
  };

  let tokenFaults = 0;
  let tokenFault: Parameters<FakeGalaxy["failTokenPolls"]>[1] = { reset: true };
  let metadataOverrides: Record<string, unknown> = {};
  let usersCurrentStatus: number | null = null;
  let usersCurrentHang = false;

  const metadata = () => ({
    issuer: metadataMode === "wrong-issuer" ? "https://not-this-galaxy.example" : baseUrl,
    ...(deviceFlow
      ? { device_authorization_endpoint: `${baseUrl}/oauth/device_authorization` }
      : {}),
    token_endpoint: `${baseUrl}/oauth/token`,
    ...(revocation ? { revocation_endpoint: `${baseUrl}/oauth/revoke` } : {}),
    grant_types_supported: deviceFlow ? [DEVICE_GRANT] : [],
    scopes_supported: ["read", "write"],
    token_endpoint_auth_methods_supported: ["none"],
    ...metadataOverrides,
  });

  const mint = (a: Approval): string => {
    const expiresIn = a.expiresIn === undefined ? 3600 : a.expiresIn;
    const token = a.accessToken ?? `fake-access-token-${tokens.size + 1}`;
    tokens.set(token, {
      token,
      username: a.username ?? "fake-user",
      email: a.email ?? "fake-user@example.org",
      admin: a.admin ?? false,
      expiresAtMs: expiresIn === null ? null : Date.now() + expiresIn * 1000,
      revoked: false,
    });
    return token;
  };

  const tokenIsValid = (t: TokenEntry | undefined): t is TokenEntry =>
    !!t && !t.revoked && (t.expiresAtMs === null || t.expiresAtMs > Date.now());

  const handle = async (req: IncomingMessage, res: ServerResponse) => {
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    const body = Buffer.concat(chunks).toString("utf8");
    const form: Record<string, string> = {};
    if ((req.headers["content-type"] ?? "").includes("application/x-www-form-urlencoded")) {
      for (const [k, v] of new URLSearchParams(body)) form[k] = v;
    }
    requests.push({
      method: req.method ?? "GET",
      url: req.url ?? "",
      headers: { ...req.headers },
      body,
      form,
    });

    const path = new URL(req.url ?? "/", "http://x").pathname;
    const method = req.method ?? "GET";

    for (const [suffix, r] of redirects) {
      if (path.endsWith(suffix)) {
        res.writeHead(r.status, { location: r.location });
        return void res.end();
      }
    }

    if (method === "GET" && path === `/.well-known/oauth-authorization-server${prefix}`) {
      switch (metadataMode) {
        case "not-found":
          // As a real Galaxy 26.1 answers: an HTML 404 page, not JSON.
          res.writeHead(404, { "content-type": "text/html" });
          return void res.end("<html><body>404 Not Found</body></html>");
        case "not-json":
          res.writeHead(200, { "content-type": "text/html" });
          return void res.end("<html>Galaxy</html>");
        case "unavailable":
          res.writeHead(503, { "content-type": "text/html" });
          return void res.end("<html>Service Unavailable</html>");
        case "malformed":
          res.writeHead(200, { "content-type": "application/json" });
          return void res.end("{not json");
        case "ok":
        case "wrong-issuer": {
          json(res, 200, metadata());
          return;
        }
      }
    }
    if (!path.startsWith(prefix + "/") && path !== prefix) {
      json(res, 404, { err_msg: "Not found" });
      return;
    }
    const local = path.slice(prefix.length);

    if (method === "POST" && local === "/oauth/device_authorization") {
      if (!deviceFlow) {
        json(res, 404, { err_msg: "Not found" });
        return;
      }
      if (deviceOverride) {
        raw(res, deviceOverride);
        return;
      }
      if (form.client_id !== knownClientId) {
        json(res, 401, { error: "invalid_client" });
        return;
      }
      counter += 1;
      const entry: DeviceEntry = {
        userCode: `FAKE-${String(counter).padStart(4, "0")}`,
        deviceCode: `fake-device-code-${counter}`,
        expiresAtMs: Date.now() + deviceExpiresIn * 1000,
        state: "pending",
        consumed: false,
      };
      devices.set(entry.userCode, entry);
      json(res, 200, {
        device_code: entry.deviceCode,
        user_code: entry.userCode,
        verification_uri: `${baseUrl}/oauth/device`,
        expires_in: deviceExpiresIn,
        interval,
      });
      return;
    }

    if (method === "POST" && local === "/oauth/token") {
      if (tokenFaults > 0) {
        tokenFaults -= 1;
        if ("reset" in tokenFault) return void req.socket.destroy();
        if ("hang" in tokenFault) return; // no answer: the client's own signal must end the request
        res.writeHead(tokenFault.status, { "content-type": "text/html", ...tokenFault.headers });
        return void res.end(tokenFault.body ?? "<html>fault</html>");
      }
      // RFC 6749 §3.2.1: a public client identifies itself on every request to the token endpoint.
      if (form.client_id !== knownClientId) {
        json(res, 401, { error: "invalid_client" });
        return;
      }
      if (form.grant_type !== DEVICE_GRANT) {
        json(res, 400, { error: "unsupported_grant_type" });
        return;
      }
      const entry = [...devices.values()].find((d) => d.deviceCode === form.device_code);
      if (!entry) {
        json(res, 400, { error: "invalid_grant" });
        return;
      }
      if (slowDowns > 0) {
        slowDowns -= 1;
        json(res, 400, { error: "slow_down" });
        return;
      }
      if (options.enforceInterval) {
        const now = Date.now();
        const tooSoon = lastPollMs !== undefined && now - lastPollMs < interval * 1000 * 0.9;
        lastPollMs = now;
        if (tooSoon) {
          tooFast += 1;
          json(res, 400, { error: "slow_down" });
          return;
        }
      }
      if (entry.state === "pending" && Date.now() >= entry.expiresAtMs) entry.state = "expired";
      if (entry.state === "pending") {
        if (tokenOverride) {
          raw(res, tokenOverride);
          return;
        }
        json(res, 400, { error: "authorization_pending" });
        return;
      }
      if (entry.state === "denied") {
        json(res, 400, { error: "access_denied" });
        return;
      }
      if (entry.state === "expired") {
        json(res, 400, { error: "expired_token" });
        return;
      }
      if (entry.consumed) {
        json(res, 400, { error: "invalid_grant" });
        return;
      }
      entry.consumed = true;
      const a = entry.approval ?? {};
      const expiresIn = a.expiresIn === undefined ? 3600 : a.expiresIn;
      const token = mint(a);
      json(res, 200, {
        access_token: token,
        token_type: "Bearer",
        ...(expiresIn === null ? {} : { expires_in: expiresIn }),
        ...(a.scopes === undefined ? {} : { scope: a.scopes.join(" ") }),
      });
      return;
    }

    if (method === "POST" && local === "/oauth/revoke") {
      if (!revocation) {
        json(res, 404, { err_msg: "Not found" });
        return;
      }
      if (form.client_id !== knownClientId) {
        json(res, 401, { error: "invalid_client" });
        return;
      }
      // RFC 7009 §2.1: `token` is required.
      if (!form.token) {
        json(res, 400, { error: "invalid_request" });
        return;
      }
      const t = tokens.get(form.token);
      if (t) t.revoked = true;
      // RFC 7009 §2.2: 200 even for tokens the server does not know.
      res.writeHead(200, { "content-length": "0" });
      return void res.end();
    }

    if (method === "GET" && local === "/api/version") {
      if (versionMode === "redirect")
        return void (res.writeHead(302, { location: "https://elsewhere.example/api/version" }),
        res.end());
      if (versionMode === "html") {
        res.writeHead(200, { "content-type": "text/html" });
        return void res.end("<html>Please log in</html>");
      }
      json(res, 200, { version_major: "26.1", version_minor: "0" });
      return;
    }

    if (method === "GET" && local === "/api/users/current") {
      if (usersCurrentHang) return;
      if (usersCurrentStatus !== null) {
        json(res, usersCurrentStatus, {});
        return;
      }
      const key = req.headers["x-api-key"];
      const t = typeof key === "string" ? tokens.get(key) : undefined;
      if (tokenIsValid(t)) {
        json(res, 200, {
          id: `fake-id-${t.username}`,
          username: t.username,
          email: t.email,
          is_admin: t.admin,
        });
        return;
      }
      if (invalidKey === "forbidden") {
        json(res, 403, {
          err_code: 403001,
          err_msg: "API authentication required for this request",
        });
        return;
      }
      json(res, 200, {
        total_disk_usage: 0,
        nice_total_disk_usage: "0 bytes",
        quota_percent: null,
      });
      return;
    }

    json(res, 404, { err_msg: "Not found" });
  };

  const server = createServer((req, res) => {
    handle(req, res).catch((err: unknown) => {
      res.writeHead(500);
      res.end(String(err));
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}${prefix}`;

  const entryFor = (userCode: string): DeviceEntry => {
    const e = devices.get(userCode);
    if (!e) throw new Error(`fake galaxy: unknown user code ${userCode}`);
    return e;
  };

  return {
    baseUrl,
    requests,
    requestsTo: (suffix) =>
      requests.filter((r) => new URL(r.url, "http://x").pathname.endsWith(suffix)),
    tooFastPolls: () => tooFast,
    deviceCodes: () => [...devices.values()].map((d) => d.deviceCode),
    pendingCodes: () =>
      [...devices.values()].filter((d) => d.state === "pending").map((d) => d.userCode),
    approve(userCode, approval = {}) {
      const e = entryFor(userCode);
      e.state = "approved";
      e.approval = { scopes: [], ...approval };
    },
    deny: (userCode) => void (entryFor(userCode).state = "denied"),
    expire: (userCode) => void (entryFor(userCode).state = "expired"),
    slowDown: (n) => void (slowDowns = n),
    setDeviceFlow: (on) => void (deviceFlow = on),
    setRevocationEndpoint: (on) => void (revocation = on),
    setMetadataMode: (m) => void (metadataMode = m),
    setVersionMode: (m) => void (versionMode = m),
    setInvalidKeyResponse: (k) => void (invalidKey = k),
    setDeviceAuthorizationOverride: (r) => void (deviceOverride = r),
    setTokenOverride: (r) => void (tokenOverride = r),
    failTokenPolls(n, fault) {
      tokenFaults = n;
      tokenFault = fault;
    },
    setMetadataOverrides: (f) => void (metadataOverrides = f ?? {}),
    setUsersCurrentStatus: (c) => void (usersCurrentStatus = c),
    hangUsersCurrent: (on) => void (usersCurrentHang = on),
    setRedirect(suffix, location, status = 302) {
      if (location === null) redirects.delete(suffix);
      else redirects.set(suffix, { location, status });
    },
    issueToken: (approval = {}) => mint(approval),
    issuedTokens: () => [...tokens.keys()],
    isTokenValid: (token) => tokenIsValid(tokens.get(token)),
    close: () =>
      new Promise<void>((resolve, reject) => {
        server.closeAllConnections();
        server.close((err) => {
          if (err) reject(err);
          else resolve();
        });
      }),
  };
}

/**
 * For a test file that needs fake Galaxies: call once at the top level. The returned function
 * starts one (as many as the test wants), and every one is closed after each test.
 */
export function useFakeGalaxies(): (options?: FakeGalaxyOptions) => Promise<FakeGalaxy> {
  const open: FakeGalaxy[] = [];
  afterEach(async () => {
    await Promise.all(open.splice(0).map((g) => g.close()));
  });
  return async (options) => {
    const g = await startFakeGalaxy(options);
    open.push(g);
    return g;
  };
}
