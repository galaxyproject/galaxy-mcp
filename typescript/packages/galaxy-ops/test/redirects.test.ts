import { createServer, type IncomingHttpHeaders, type Server, type ServerResponse } from "node:http";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createGalaxyClient } from "../src/client";
import { createGalaxyContext } from "../src/context";
import { GalaxyError, GalaxyValidationError } from "../src/errors";
import { tusUploadFile, uploadPathIn } from "../src/tus-upload";
import { getUserOp } from "../src/operations/get-user";
import { getPageOp } from "../src/operations/get-page";
import { getServerInfoOp } from "../src/operations/get-server-info";
import { uploadFileOp } from "../src/operations/upload-file";
import { runWithEnvelope } from "../src/operations/registry";

const KEY = "SECRET-KEY";

interface Hit {
  method: string;
  url: string;
  headers: IncomingHttpHeaders;
}

interface Running {
  base: string;
  port: number;
  hits: Hit[];
}

const servers: Server[] = [];

// Every server binds 127.0.0.1; two of them are different origins because their ports differ.
async function listen(handler: (req: Hit, res: ServerResponse) => void): Promise<Running> {
  const hits: Hit[] = [];
  const server = createServer((req, res) => {
    const hit = { method: req.method ?? "", url: req.url ?? "", headers: req.headers };
    hits.push(hit);
    req.resume();
    req.on("end", () => handler(hit, res));
  });
  servers.push(server);
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return { base: `http://127.0.0.1:${port}`, port, hits };
}

afterEach(async () => {
  await Promise.all(servers.splice(0).map((s) => new Promise((r) => s.close(r))));
});

const getUser = (base: string) =>
  createGalaxyClient(base, KEY).GET("/api/users/{user_id}", {
    params: { path: { user_id: "current" } },
  });

const caught = (p: Promise<unknown>) =>
  p.then(
    () => undefined,
    (e: unknown) => e,
  );

describe("redirects are never followed (two real servers)", () => {
  for (const status of [301, 302, 303, 307, 308]) {
    it(`${status} to another origin`, async () => {
      const target = await listen((_hit, res) => res.end("{}"));
      const galaxy = await listen((_hit, res) => {
        res.writeHead(status, { location: `${target.base}/steal` });
        res.end();
      });
      const err = await caught(getUser(galaxy.base));
      expect(err).toBeInstanceOf(GalaxyValidationError);
      expect((err as GalaxyError).message).toBe(
        `Galaxy answered ${status} and redirects are not followed; ` +
          `Galaxy redirected to ${target.base}/steal -- check GALAXY_URL`,
      );
      expect((err as GalaxyError).http?.status).toBe(status);
      expect(galaxy.hits).toHaveLength(1);
      expect(target.hits).toHaveLength(0);
    });
  }

  it("a protocol-relative Location", async () => {
    const target = await listen((_hit, res) => res.end("{}"));
    const galaxy = await listen((_hit, res) => {
      res.writeHead(302, { location: `//127.0.0.1:${target.port}/steal` });
      res.end();
    });
    await expect(getUser(galaxy.base)).rejects.toThrow(`redirected to ${target.base}/steal`);
    expect(galaxy.hits).toHaveLength(1);
    expect(target.hits).toHaveLength(0);
  });

  it("a same-origin 302 is not followed either, so its off-origin second hop never happens", async () => {
    const target = await listen((_hit, res) => res.end("{}"));
    const galaxy = await listen((hit, res) => {
      res.writeHead(302, {
        location: hit.url.startsWith("/api/users/") ? "/hop" : `${target.base}/steal`,
      });
      res.end();
    });
    await expect(getUser(galaxy.base)).rejects.toThrow(`redirected to ${galaxy.base}/hop`);
    expect(galaxy.hits.map((h) => h.url)).toEqual(["/api/users/current"]);
    expect(target.hits).toHaveLength(0);
  });

  it("a POST answered 307 is not re-sent", async () => {
    const target = await listen((_hit, res) => res.end("{}"));
    const galaxy = await listen((_hit, res) => {
      res.writeHead(307, { location: `${target.base}/steal` });
      res.end("moved");
    });
    const client = createGalaxyClient(galaxy.base, KEY);
    await expect(
      client.POST("/api/histories", { body: { name: "x" } as never }),
    ).rejects.toMatchObject({ http: { status: 307, method: "POST", bodyText: "moved" } });
    expect(galaxy.hits).toHaveLength(1);
    expect(target.hits).toHaveLength(0);
  });
});

describe("redirects are never followed (mock fetch)", () => {
  const ok = () => new Response("{}", { status: 200 });
  const user = { params: { path: { user_id: "current" } } } as const;

  it("every request goes out with redirect: manual", async () => {
    const fetchSpy = vi.fn(async (_req: Request) => ok());
    await createGalaxyClient("https://galaxy.example", KEY, fetchSpy).GET("/api/users/{user_id}", user);
    expect(fetchSpy.mock.calls[0]![0].redirect).toBe("manual");
  });

  it("a call cannot opt back into following", async () => {
    const fetchSpy = vi.fn(async () => ok());
    const client = createGalaxyClient("https://galaxy.example", KEY, fetchSpy);
    await expect(client.GET("/api/users/{user_id}", { ...user, redirect: "follow" })).rejects.toThrow(
      "would follow redirects",
    );
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("a call cannot substitute its own fetch", async () => {
    const fetchSpy = vi.fn(async () => ok());
    const sneaky = vi.fn(async () => ok());
    const client = createGalaxyClient("https://galaxy.example", KEY, fetchSpy);
    await expect(client.GET("/api/users/{user_id}", { ...user, fetch: sneaky })).rejects.toThrow(
      "replaces the client's fetch",
    );
    expect(sneaky).not.toHaveBeenCalled();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("a call cannot substitute its own Request class", async () => {
    const fetchSpy = vi.fn(async () => ok());
    class Sneaky extends Request {}
    const client = createGalaxyClient("https://galaxy.example", KEY, fetchSpy);
    await expect(
      client.GET("/api/users/{user_id}", { ...user, Request: Sneaky } as never),
    ).rejects.toThrow("replaces the client's fetch or Request");
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("a call cannot point the key at another baseUrl", async () => {
    const fetchSpy = vi.fn(async () => ok());
    const client = createGalaxyClient("https://galaxy.example/galaxy", KEY, fetchSpy);
    for (const baseUrl of ["https://elsewhere.example", "https://galaxy.example/other"]) {
      await expect(client.GET("/api/users/{user_id}", { ...user, baseUrl })).rejects.toThrow(
        "refusing to send the API key outside",
      );
    }
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("a 3xx with a body is an error, never content", async () => {
    const fetchSpy = vi.fn(
      async () =>
        new Response(JSON.stringify({ id: "1" }), {
          status: 302,
          headers: { location: "https://elsewhere.example/", "content-type": "application/json" },
        }),
    );
    const client = createGalaxyClient("https://galaxy.example", KEY, fetchSpy);
    await expect(client.GET("/api/users/{user_id}", user)).rejects.toThrow(
      "Galaxy redirected to https://elsewhere.example/",
    );
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });

  it("a browser's opaque redirect is refused too", async () => {
    const opaque = { type: "opaqueredirect", status: 0, ok: false, headers: new Headers() };
    const client = createGalaxyClient("https://galaxy.example", KEY, async () => opaque as Response);
    const err = await caught(client.GET("/api/users/{user_id}", user));
    expect(err).toBeInstanceOf(GalaxyValidationError);
    expect((err as Error).message).toBe(
      "Galaxy answered with a redirect and redirects are not followed -- check GALAXY_URL",
    );
  });

  it("shows where a redirect pointed without userinfo or query, and capped", async () => {
    const loc = "https://user:pw@elsewhere.example/x?X-Amz-Signature=abc#frag";
    const client = createGalaxyClient("https://galaxy.example", KEY, async () =>
      new Response(null, { status: 302, headers: { location: loc } }),
    );
    const err = (await caught(client.GET("/api/users/{user_id}", user))) as GalaxyError;
    expect(err.message).toContain("Galaxy redirected to https://elsewhere.example/x -- check");
    expect(err.http?.redirectedTo).toBe("https://elsewhere.example/x");
    for (const text of [err.message, JSON.stringify(err.http)]) {
      expect(text).not.toContain("pw");
      expect(text).not.toContain("Signature");
      expect(text).not.toContain("frag");
    }

    const long = `https://elsewhere.example/${"a".repeat(500)}`;
    const client2 = createGalaxyClient("https://galaxy.example", KEY, async () =>
      new Response(null, { status: 302, headers: { location: long } }),
    );
    const err2 = (await caught(client2.GET("/api/users/{user_id}", user))) as GalaxyError;
    expect(err2.http?.redirectedTo).toBe(`${long.slice(0, 200)}...`);
  });

  it("a 304 is a plain non-2xx, not a redirect", async () => {
    const fetchSpy = vi.fn(async () => new Response(null, { status: 304 }));
    const client = createGalaxyClient("https://galaxy.example", KEY, fetchSpy);
    const { data, response } = await client.GET("/api/users/{user_id}", user);
    expect(response.status).toBe(304);
    expect(data).toBeUndefined();
    expect(fetchSpy).toHaveBeenCalledTimes(1);
  });
});

describe("an unfollowed 3xx on the surfaces", () => {
  const ctxAnswering = (status: number, location: string, body = "") =>
    createGalaxyContext({
      baseUrl: "http://usegalaxy.example",
      apiKey: KEY,
      serverVersion: "26.0",
      fetchImpl: (async () =>
        new Response(body || null, { status, headers: { location } })) as unknown as typeof fetch,
    });

  it("a proxy's 301 http -> https names the https Location, with a non-retryable kind", async () => {
    const ctx = ctxAnswering(
      301,
      "https://usegalaxy.example/api/users/current",
      "<html><body>301 Moved Permanently</body></html>",
    );
    const env = await runWithEnvelope(getUserOp, {}, ctx);
    expect(env).toMatchObject({ success: false, errorKind: "validation" });
    expect(env.message).toBe(
      "Failed to get user: Galaxy answered 301 and redirects are not followed; " +
        "Galaxy redirected to https://usegalaxy.example/api/users/current -- check GALAXY_URL",
    );
  });

  it("a raise-for-status op says the same, never requests' impossible '302 Client Error'", async () => {
    const ctx = ctxAnswering(302, "https://elsewhere.example/");
    const env = await runWithEnvelope(getPageOp, { pageId: "p1" } as never, ctx);
    expect(env.errorKind).toBe("validation");
    expect(env.message).toContain(
      "Galaxy answered 302 and redirects are not followed; Galaxy redirected to https://elsewhere.example/",
    );
    expect(env.message).not.toContain("Client Error");
  });

  it("get_server_info keeps the facts of a version probe answered with a redirect", async () => {
    const ctx = createGalaxyContext({
      baseUrl: "http://usegalaxy.example",
      apiKey: KEY,
      fetchImpl: (async () =>
        new Response(null, {
          status: 301,
          headers: { location: "https://usegalaxy.example/api/version" },
        })) as unknown as typeof fetch,
    });
    const env = await runWithEnvelope(getServerInfoOp, {} as never, ctx);
    expect(env.success).toBe(false);
    expect(env.errorKind).toBe("validation");
    expect(env.message).toContain("Galaxy redirected to https://usegalaxy.example/api/version");
  });
});

describe("tus upload: the key only goes to the configured Galaxy", () => {
  const dirs: string[] = [];
  afterEach(() => {
    for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
  });

  const file = () => {
    const dir = mkdtempSync(join(tmpdir(), "tus-redirect-"));
    dirs.push(dir);
    const path = join(dir, "f.txt");
    writeFileSync(path, "abc");
    return path;
  };

  /** A tus server whose creation reply names `location(galaxy)`. */
  async function tusGalaxy(location: (g: Running) => string) {
    let self: Running | undefined;
    const g = await listen((hit, res) => {
      if (hit.method === "POST" && hit.url === "/api/upload/resumable_upload") {
        res.writeHead(201, { "tus-resumable": "1.0.0", location: location(self!) });
      } else {
        res.writeHead(204, { "tus-resumable": "1.0.0", "upload-offset": "3" });
      }
      res.end();
    });
    self = g;
    return g;
  }

  it("an absolute same-origin Location works", async () => {
    const galaxy = await tusGalaxy((g) => `${g.base}/api/upload/resumable_upload/abc123`);
    await expect(tusUploadFile({ baseUrl: galaxy.base, apiKey: KEY, path: file() })).resolves.toBe(
      "abc123",
    );
    expect(galaxy.hits.map((h) => `${h.method} ${h.url}`)).toEqual([
      "POST /api/upload/resumable_upload",
      "PATCH /api/upload/resumable_upload/abc123",
    ]);
  });

  for (const [name, loc] of [
    ["another host and port", (t: Running) => `${t.base}/api/upload/resumable_upload/abc123`],
    ["the same host, another scheme", (t: Running) => `https://127.0.0.1:${t.port}/api/upload/resumable_upload/abc123`],
  ] as const) {
    it(`an absolute Location on ${name} is re-rooted onto the configured origin`, async () => {
      const target = await listen((_hit, res) => res.end());
      const galaxy = await tusGalaxy(() => loc(target));
      await expect(tusUploadFile({ baseUrl: galaxy.base, apiKey: KEY, path: file() })).resolves.toBe(
        "abc123",
      );
      expect(target.hits).toHaveLength(0);
      expect(galaxy.hits.map((h) => `${h.method} ${h.url} ${h.headers["x-api-key"]}`)).toEqual([
        `POST /api/upload/resumable_upload ${KEY}`,
        `PATCH /api/upload/resumable_upload/abc123 ${KEY}`,
      ]);
    });
  }

  it("keeps the Location's query and drops its fragment", async () => {
    const target = await listen((_hit, res) => res.end());
    const galaxy = await tusGalaxy(() => `${target.base}/api/upload/resumable_upload/abc123?x=1#frag`);
    // The session id is the upload's last path segment -- the query is not part of it.
    await expect(tusUploadFile({ baseUrl: galaxy.base, apiKey: KEY, path: file() })).resolves.toBe(
      "abc123",
    );
    expect(galaxy.hits[1]!.url).toBe("/api/upload/resumable_upload/abc123?x=1");
    expect(target.hits).toHaveLength(0);
  });

  it("re-roots under a Galaxy served from a sub-path", async () => {
    const target = await listen((_hit, res) => res.end());
    let self: Running | undefined;
    const galaxy = await listen((hit, res) => {
      if (hit.method === "POST") {
        res.writeHead(201, {
          "tus-resumable": "1.0.0",
          location: `${target.base}/galaxy/api/upload/resumable_upload/abc123`,
        });
      } else {
        res.writeHead(204, { "tus-resumable": "1.0.0", "upload-offset": "3" });
      }
      res.end();
    });
    self = galaxy;
    await tusUploadFile({ baseUrl: `${self.base}/galaxy/`, apiKey: KEY, path: file() });
    expect(galaxy.hits.map((h) => h.url)).toEqual([
      "/galaxy/api/upload/resumable_upload",
      "/galaxy/api/upload/resumable_upload/abc123",
    ]);
    expect(target.hits).toHaveLength(0);
  });

  it("refuses a Location off the upload route once -- no retries -- inside the envelope", async () => {
    const target = await listen((_hit, res) => res.end());
    const galaxy = await tusGalaxy(() => `${target.base}/api/histories/steal`);
    const ctx = createGalaxyContext({ baseUrl: galaxy.base, apiKey: KEY, serverVersion: "26.0" });
    const started = Date.now();
    const env = await runWithEnvelope(uploadFileOp, { path: file() } as never, ctx);
    expect(Date.now() - started).toBeLessThan(900); // the first tus retry waits 1s
    expect(env).toMatchObject({ success: false, errorKind: "connection" });
    expect(env.message).toBe(
      `Failed to upload file: tus upload refused: Galaxy's upload Location ${target.base}/api/histories/steal ` +
        "is not its resumable-upload route, so the API key was not sent to it",
    );
    expect(galaxy.hits.map((h) => `${h.method} ${h.url}`)).toEqual(["POST /api/upload/resumable_upload"]);
    expect(target.hits).toHaveLength(0);
  });

  /** A tus server that answers `answer(hit)` -- the tests below each break one reply. */
  async function tusAnswering(answer: (hit: Hit, g: Running) => [number, Record<string, string>]) {
    let self: Running | undefined;
    const g = await listen((hit, res) => {
      const [status, headers] = answer(hit, self!);
      res.writeHead(status, { "tus-resumable": "1.0.0", ...headers });
      res.end();
    });
    self = g;
    return g;
  }

  const uploadEnvelope = async (g: Running) => {
    const ctx = createGalaxyContext({ baseUrl: g.base, apiKey: KEY, serverVersion: "26.0" });
    const started = Date.now();
    const env = await runWithEnvelope(uploadFileOp, { path: file() } as never, ctx);
    expect(Date.now() - started).toBeLessThan(900); // the first tus retry would wait 1s
    return env;
  };

  it("a proxy's 301 on the creation POST is the redirect refusal, not a bad Location", async () => {
    const galaxy = await tusAnswering((_hit, g) => [
      301,
      { location: `https://127.0.0.1:${g.port}/api/upload/resumable_upload` },
    ]);
    const env = await uploadEnvelope(galaxy);
    expect(env).toMatchObject({ success: false, errorKind: "validation" });
    expect(env.message).toBe(
      "Failed to upload file: Galaxy answered 301 and redirects are not followed; Galaxy " +
        `redirected to https://127.0.0.1:${galaxy.port}/api/upload/resumable_upload -- check GALAXY_URL`,
    );
    expect(galaxy.hits).toHaveLength(1);
  });

  it("a 307 naming a valid upload path is still a redirect, refused once", async () => {
    const galaxy = await tusAnswering((_hit, g) => [
      307,
      { location: `${g.base}/api/upload/resumable_upload/abc123` },
    ]);
    const env = await uploadEnvelope(galaxy);
    expect(env).toMatchObject({ success: false, errorKind: "validation" });
    expect(env.message).toContain("Galaxy answered 307 and redirects are not followed");
    expect(galaxy.hits).toHaveLength(1);
  });

  it("a 3xx on the PATCH is refused the same way, without a retry", async () => {
    const galaxy = await tusAnswering((hit, g) =>
      hit.method === "POST"
        ? [201, { location: `${g.base}/api/upload/resumable_upload/abc123` }]
        : [307, { location: `${g.base}/api/upload/resumable_upload/abc123` }],
    );
    const env = await uploadEnvelope(galaxy);
    expect(env).toMatchObject({ success: false, errorKind: "validation" });
    expect(env.message).toContain("Galaxy answered 307 and redirects are not followed");
    expect(galaxy.hits.map((h) => h.method)).toEqual(["POST", "PATCH"]);
  });

  it("a rejected key on the upload is auth, with tus's detail kept", async () => {
    const galaxy = await tusAnswering(() => [403, {}]);
    const env = await uploadEnvelope(galaxy);
    expect(env).toMatchObject({ success: false, errorKind: "auth" });
    expect(env.message).toMatch(/^Failed to upload file: tus: unexpected response while creating upload/);
    expect(galaxy.hits).toHaveLength(1);
  });

  it("refuses a traversal in the raw Location rather than collapsing it", async () => {
    const target = await listen((_hit, res) => res.end());
    const galaxy = await tusGalaxy(() => `${target.base}/api/upload/resumable_upload/../../histories`);
    await expect(tusUploadFile({ baseUrl: galaxy.base, apiKey: KEY, path: file() })).rejects.toThrow(
      "tus upload refused",
    );
    expect(galaxy.hits).toHaveLength(1);
    expect(target.hits).toHaveLength(0);
  });

  describe("uploadPathIn decides on the raw Location", () => {
    const prefix = "/api/upload/resumable_upload/";
    it.each([
      ["http://other:1/api/upload/resumable_upload/abc", "/api/upload/resumable_upload/abc"],
      ["//other/api/upload/resumable_upload/abc?x=1#f", "/api/upload/resumable_upload/abc?x=1"],
      ["/api/upload/resumable_upload/abc", "/api/upload/resumable_upload/abc"],
    ])("keeps %s", (loc, kept) => expect(uploadPathIn(loc, prefix)).toBe(kept));
    it.each([
      "http://other/api/upload/resumable_upload/../x",
      "http://other/api/upload/resumable_upload/%2e%2e/x",
      "http://other/api/upload/resumable_upload/%2E%2E",
      "http://other/api/upload/resumable_upload/..%2fx",
      "http://other/api/upload/resumable_upload//x",
      "http://other//api/upload/resumable_upload/x",
      "http://other/api/upload/resumable_upload/a\\..\\x",
      "http://other\\api/upload/resumable_upload/x",
      "http://other/api/upload/resumable_upload/",
      "http://other/api/upload/resumable_upload/a/b",
      "http://other/api/histories/x",
      "api/upload/resumable_upload/x",
    ])("refuses %s", (loc) => expect(uploadPathIn(loc, prefix)).toBeUndefined());
  });
});
