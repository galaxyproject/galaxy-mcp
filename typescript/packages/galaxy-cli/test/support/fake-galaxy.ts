/**
 * A fake Galaxy for testing `gxy auth`: `GET /api/version` and `GET /api/users/current` on a
 * `node:http` server bound to 127.0.0.1 with an OS-assigned port.
 *
 * What is real and what is a guess. Confirmed against a real Galaxy 26.1: an unknown key at
 * `/api/users/current` gets 200 with an anonymous record that has no `username`, and
 * `/api/version` is JSON with `version_major`. Everything else is an assumption about what Galaxy
 * will do, not an observation: the exact fields of the anonymous record and of a user record, the
 * 403 shape of `setInvalidKeyResponse("forbidden")`, and the redirect and error faults. Tests
 * should assert the behaviour of our code against these, not the literals.
 *
 * - `/api/users/current` knows a user only by a token this server issued
 * - `/api/version` answers the version, or a redirect or an HTML page for the checks that run
 *   before a key is sent anywhere
 *
 * It is a real HTTP server, not a stubbed `fetch`, because the redirects, connection resets and
 * timing that the client must survive are only real when there is a socket.
 *
 * Everything a test needs to steer is a method on the returned object, and every request is
 * recorded in `requests`. Tokens are obviously fake (`fake-access-token-N`).
 *
 *   const galaxy = await startFakeGalaxy();
 *   const token = galaxy.issueToken({ admin: true });
 *   await galaxy.close();
 */
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach } from "vitest";

export interface Approval {
  username?: string;
  email?: string;
  /** Make `/api/users/current` report `is_admin: true` for this token. */
  admin?: boolean;
}

/** One request as the server received it. */
export interface RecordedRequest {
  method: string;
  /** Path and query as sent. */
  url: string;
  /** Lower-cased header names. */
  headers: Record<string, string | string[] | undefined>;
}

/**
 * How `/api/version` answers, for the check that runs before a token is sent anywhere: the
 * version (`ok`), a 302 to another origin (`redirect`, as a wrong address such as http for
 * https or a missing path prefix often answers), or a 200 HTML page (`html`, as a Galaxy behind
 * a `remote_user` proxy answers). The last two are the answers a token must never follow.
 */
export type VersionMode = "ok" | "redirect" | "html";

export interface FakeGalaxy {
  /** `http://127.0.0.1:<port>`: the Galaxy base URL. */
  readonly baseUrl: string;
  readonly requests: RecordedRequest[];
  /** Requests whose URL ends with `suffix` (a path such as `/api/version`). */
  requestsTo(suffix: string): RecordedRequest[];
  /** What `/api/version` answers: the version (default), a 302 to another origin, or an HTML page. */
  setVersionMode(mode: VersionMode): void;
  /** What `/api/users/current` says for an unknown key: 200 with the anonymous shape (default, observed), or 403 (a guess). */
  setInvalidKeyResponse(kind: "anonymous" | "forbidden"): void;
  /**
   * Answer every request whose path ends with `pathSuffix` with a redirect to `location`
   * (default 302), before anything else; `null` removes it. For proving a client does not
   * follow a redirect with credentials attached.
   */
  setRedirect(pathSuffix: string, location: string | null, status?: number): void;
  /** Make `/api/users/current` answer this status with an empty JSON body; `null` restores it. */
  setUsersCurrentStatus(status: number | null): void;
  /** Issue a token `/api/users/current` will know. */
  issueToken(options?: Approval): string;
  close(): Promise<void>;
}

interface TokenEntry {
  username: string;
  email: string;
  admin: boolean;
}

export async function startFakeGalaxy(): Promise<FakeGalaxy> {
  let versionMode: VersionMode = "ok";
  let invalidKey: "anonymous" | "forbidden" = "anonymous";
  const redirects = new Map<string, { location: string; status: number }>();

  const requests: RecordedRequest[] = [];
  const tokens = new Map<string, TokenEntry>();
  let baseUrl = "";

  const json = (res: ServerResponse, status: number, body: unknown) => {
    res.writeHead(status, { "content-type": "application/json", "cache-control": "no-store" });
    res.end(JSON.stringify(body));
  };

  let usersCurrentStatus: number | null = null;

  const mint = (a: Approval): string => {
    const token = `fake-access-token-${tokens.size + 1}`;
    tokens.set(token, {
      username: a.username ?? "fake-user",
      email: a.email ?? "fake-user@example.org",
      admin: a.admin ?? false,
    });
    return token;
  };

  const handle = (req: IncomingMessage, res: ServerResponse) => {
    requests.push({
      method: req.method ?? "GET",
      url: req.url ?? "",
      headers: { ...req.headers },
    });

    const path = new URL(req.url ?? "/", "http://x").pathname;
    const method = req.method ?? "GET";

    for (const [suffix, r] of redirects) {
      if (path.endsWith(suffix)) {
        res.writeHead(r.status, { location: r.location });
        return void res.end();
      }
    }

    if (method === "GET" && path === "/api/version") {
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

    if (method === "GET" && path === "/api/users/current") {
      if (usersCurrentStatus !== null) {
        json(res, usersCurrentStatus, {});
        return;
      }
      const key = req.headers["x-api-key"];
      const t = typeof key === "string" ? tokens.get(key) : undefined;
      if (t) {
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
    try {
      handle(req, res);
    } catch (err) {
      res.writeHead(500);
      res.end(String(err));
    }
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;

  return {
    baseUrl,
    requests,
    requestsTo: (suffix) =>
      requests.filter((r) => new URL(r.url, "http://x").pathname.endsWith(suffix)),
    setVersionMode: (m) => void (versionMode = m),
    setInvalidKeyResponse: (k) => void (invalidKey = k),
    setUsersCurrentStatus: (c) => void (usersCurrentStatus = c),
    setRedirect(suffix, location, status = 302) {
      if (location === null) redirects.delete(suffix);
      else redirects.set(suffix, { location, status });
    },
    issueToken: (approval = {}) => mint(approval),
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
export function useFakeGalaxies(): () => Promise<FakeGalaxy> {
  const open: FakeGalaxy[] = [];
  afterEach(async () => {
    await Promise.all(open.splice(0).map((g) => g.close()));
  });
  return async () => {
    const g = await startFakeGalaxy();
    open.push(g);
    return g;
  };
}
