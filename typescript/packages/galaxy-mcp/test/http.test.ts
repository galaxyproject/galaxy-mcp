import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { request, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { createGalaxyContext } from "@galaxyproject/galaxy-ops";
import {
  checkHttpStartup,
  hostAllowed,
  httpConfig,
  HttpStartupError,
  isLoopbackHost,
  originAllowed,
  startHttpServer,
} from "../src/http";
import { buildServer, toolNames } from "../src/server";

describe("loopback hosts", () => {
  it.each(["127.0.0.1", "127.8.9.10", "localhost", "LOCALHOST.", "::1", "[::1]", "0:0:0:0:0:0:0:1"])(
    "treats %s as loopback",
    (host) => expect(isLoopbackHost(host)).toBe(true),
  );
  it.each(["0.0.0.0", "::", "10.0.0.1", "example.org", "localhost.example.org", ""])(
    "does not treat %s as loopback",
    (host) => expect(isLoopbackHost(host)).toBe(false),
  );
});

describe("HTTP startup", () => {
  it("serves on loopback without being asked", () => {
    expect(() => checkHttpStartup("127.0.0.1", false)).not.toThrow();
  });
  it("refuses a routable address unless told something else controls access", () => {
    expect(() => checkHttpStartup("0.0.0.0", false)).toThrow(HttpStartupError);
    expect(() => checkHttpStartup("0.0.0.0", false)).toThrow(/--allow-unauthenticated/);
    expect(() => checkHttpStartup("0.0.0.0", true)).not.toThrow();
  });
});

describe("HTTP configuration", () => {
  it("uses the Python server's defaults", () => {
    expect(httpConfig({}, {})).toEqual({
      host: "127.0.0.1",
      port: 8000,
      path: "/",
      allowedHosts: new Set(),
      allowedOrigins: new Set(),
      allowLocalFiles: false,
    });
  });

  it("reads the environment, and lets a flag win over it", () => {
    const env = {
      GALAXY_MCP_HOST: "0.0.0.0",
      GALAXY_MCP_PORT: "9000",
      GALAXY_MCP_HTTP_PATH: "mcp",
      GALAXY_MCP_ALLOW_UNAUTHENTICATED: "true",
      GALAXY_MCP_ALLOWED_HOSTS: "MCP.example.org, other.example.org",
      GALAXY_MCP_ALLOWED_ORIGINS: "https://Client.example.org/",
      GALAXY_MCP_ALLOW_LOCAL_FILES: "1",
    };
    expect(httpConfig({}, env)).toMatchObject({
      host: "0.0.0.0",
      port: 9000,
      path: "/mcp",
      allowedHosts: new Set(["mcp.example.org", "other.example.org"]),
      allowedOrigins: new Set(["https://client.example.org"]),
      allowLocalFiles: true,
    });
    expect(httpConfig({ host: "127.0.0.1", port: "8123", path: "/x" }, env)).toMatchObject({
      host: "127.0.0.1",
      port: 8123,
      path: "/x",
    });
  });

  it("refuses a port that is not one, and an unauthenticated routable address", () => {
    expect(() => httpConfig({ port: "80a" }, {})).toThrow("invalid port: 80a");
    expect(() => httpConfig({}, { GALAXY_MCP_PORT: "" })).toThrow("invalid port: ");
    expect(() => httpConfig({ host: "0.0.0.0" }, {})).toThrow(HttpStartupError);
    expect(httpConfig({ host: "0.0.0.0", allowUnauthenticated: true }, {}).host).toBe("0.0.0.0");
  });
});

describe("Host and Origin", () => {
  const guards = { allowedHosts: new Set(["mcp.example.org"]), allowedOrigins: new Set(["https://client.example.org"]) };

  it("answers a loopback Host or one it was told about", () => {
    expect(hostAllowed("127.0.0.1:8000", guards)).toBe(true);
    expect(hostAllowed("[::1]:8000", guards)).toBe(true);
    expect(hostAllowed("MCP.example.org", guards)).toBe(true);
    expect(hostAllowed("attacker.example", guards)).toBe(false);
    expect(hostAllowed("anything", { ...guards, allowedHosts: new Set(["*"]) })).toBe(true);
  });

  it("answers pages from this machine or the allowlist", () => {
    expect(originAllowed("http://localhost:6274", guards)).toBe(true);
    expect(originAllowed("https://client.example.org/", guards)).toBe(true);
    expect(originAllowed("https://evil.example", guards)).toBe(false);
    expect(originAllowed("https://evil.example", { ...guards, allowedOrigins: new Set(["*"]) })).toBe(true);
  });
});

/** Reads a whole response; one cut off partway is a failure, not a pass on whatever made it through. */
function collect(res: IncomingMessage): Promise<{ status: number; headers: IncomingMessage["headers"]; body: string }> {
  return new Promise((resolve, reject) => {
    let body = "";
    res.setEncoding("utf8");
    res.on("data", (chunk) => (body += chunk));
    res.on("end", () => resolve({ status: res.statusCode ?? 0, headers: res.headers, body }));
    res.on("error", reject);
    res.on("close", () => {
      if (!res.complete) reject(new Error("response closed before it was complete"));
    });
  });
}

/** A raw request, so the test controls the Host header that fetch would not let it set. */
function raw(
  port: number,
  {
    method = "POST",
    path = "/mcp",
    headers = {},
    body,
  }: { method?: string; path?: string; headers?: Record<string, string>; body?: string | string[] },
): Promise<{ status: number; headers: Record<string, string | string[] | undefined>; body: string }> {
  return new Promise((resolve, reject) => {
    const req = request({ host: "127.0.0.1", port, method, path, headers }, (res) => collect(res).then(resolve, reject));
    req.on("error", reject);
    // An array goes out piece by piece with chunked encoding, so the server never sees a length.
    if (Array.isArray(body)) {
      for (const chunk of body) req.write(chunk);
      req.end();
    } else {
      req.end(body);
    }
  });
}

describe("the Streamable HTTP server", () => {
  let server: Server;
  let port: number;
  const url = () => new URL(`http://127.0.0.1:${port}/mcp`);
  const initialize = JSON.stringify({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "t", version: "0" } },
  });
  const mcpHeaders = { "content-type": "application/json", accept: "application/json, text/event-stream" };

  beforeAll(async () => {
    // Nothing here reaches Galaxy: a call that gets that far fails at this fetch.
    const ctx = createGalaxyContext({
      baseUrl: "http://galaxy.invalid",
      apiKey: "k",
      fetchImpl: async () => {
        throw new Error("no Galaxy in this test");
      },
    });
    server = await startHttpServer({
      host: "127.0.0.1",
      port: 0,
      path: "/mcp",
      allowedHosts: new Set(["mcp.example.org"]),
      allowedOrigins: new Set(["https://client.example.org"]),
      newServer: () => buildServer(ctx),
    });
    port = (server.address() as AddressInfo).port;
  });
  afterAll(() => new Promise<void>((done) => server.close(() => done())));

  async function connect(): Promise<Client> {
    const client = new Client({ name: "http-test", version: "0" });
    await client.connect(new StreamableHTTPClientTransport(url()));
    return client;
  }

  it("lists every operation, as stdio does", async () => {
    const client = await connect();
    const { tools } = await client.listTools();
    expect(tools.map((t) => t.name).sort()).toEqual(toolNames().sort());
    await client.close();
  });

  it("answers a Host it was told about and refuses any other", async () => {
    const ok = await raw(port, { headers: { ...mcpHeaders, host: "mcp.example.org" }, body: initialize });
    expect(ok.status).toBe(200);
    const rebound = await raw(port, { headers: { ...mcpHeaders, host: "attacker.example" }, body: initialize });
    expect(rebound).toMatchObject({ status: 421, body: "Invalid Host header" });
  });

  it("answers browser pages from this machine or the allowlist, with CORS", async () => {
    for (const origin of ["http://localhost:6274", "https://client.example.org"]) {
      const res = await raw(port, { headers: { ...mcpHeaders, origin }, body: initialize });
      expect(res.status).toBe(200);
      expect(res.headers["access-control-allow-origin"]).toBe(origin);
    }
    const hostile = await raw(port, { headers: { ...mcpHeaders, origin: "https://evil.example" }, body: initialize });
    expect(hostile).toMatchObject({ status: 403, body: "Origin not allowed" });
  });

  it("answers a preflight before anything else", async () => {
    const res = await raw(port, {
      method: "OPTIONS",
      headers: { origin: "http://127.0.0.1:6274", "access-control-request-method": "POST" },
    });
    expect(res.status).toBe(204);
    expect(res.headers["access-control-allow-methods"]).toBe("POST");
  });

  it("reports a port that is already taken rather than serving on it", async () => {
    const attempt = startHttpServer({
      host: "127.0.0.1",
      port,
      path: "/mcp",
      allowedHosts: new Set(),
      allowedOrigins: new Set(),
      newServer: () => {
        throw new Error("never reached");
      },
    });
    await expect(attempt).rejects.toThrow(/EADDRINUSE/);
  });

  // The SDK reads the body through the Node-to-Web Request adapter and refuses anything over
  // 4 MiB, from Content-Length when one is sent and otherwise by counting the stream, so these
  // depend on the adapter passing that header through and streaming the body faithfully.
  describe("request bodies", () => {
    const limit = 4 * 1024 * 1024;
    const padded = (bytes: number) => {
      const message = JSON.parse(initialize);
      message.params._meta = { pad: "x".repeat(bytes) };
      return JSON.stringify(message);
    };
    const inChunks = (text: string, size = 64 * 1024) =>
      Array.from({ length: Math.ceil(text.length / size) }, (_, i) => text.slice(i * size, (i + 1) * size));

    it("reads a large body whole, streamed in chunks", async () => {
      // The id comes back verbatim, and no two stretches of it are alike, so a chunk dropped,
      // repeated or reordered on the way in shows up in the reply rather than vanishing into
      // padding the server ignores.
      let id = "";
      for (let i = 0; id.length < 3 * 1024 * 1024; i++) id += `${i},`;
      const body = JSON.stringify({ ...JSON.parse(initialize), id });
      expect(body.length).toBeLessThan(limit);
      const res = await raw(port, { headers: mcpHeaders, body: inChunks(body) });
      expect(res.status).toBe(200);
      expect(res.body.includes(`"id":${JSON.stringify(id)}`), "the id came back changed").toBe(true);
      expect(res.body).toContain('"serverInfo"');
    });

    it("takes a body of exactly the limit, sent with its length", async () => {
      // Together with the limit + 1 refusal below, this pins the cap at 4 MiB in bytes.
      const body = padded(limit - padded(0).length);
      expect(Buffer.byteLength(body)).toBe(limit);
      const headers = { ...mcpHeaders, "content-length": String(limit) };
      const res = await raw(port, { headers, body });
      expect(res.status).toBe(200);
      expect(res.body).toContain('"serverInfo"');
    });

    it("refuses an oversized body by its declared length, before reading any of it", async () => {
      // Only the headers go out: the refusal has to come from Content-Length alone, and not
      // sending the body keeps the server's close from racing a write.
      const res = await new Promise<{ status: number; body: string }>((resolve, reject) => {
        const headers = { ...mcpHeaders, "content-length": String(limit + 1) };
        let answered = false;
        // The deadline runs until the whole response is in, and every outcome tears the request
        // down, so a stalled reply fails here instead of hanging the suite.
        const settle = (outcome: () => void) => {
          clearTimeout(timer);
          req.destroy();
          outcome();
        };
        const req = request({ host: "127.0.0.1", port, method: "POST", path: "/mcp", headers }, (res) => {
          answered = true;
          collect(res).then(
            (got) => settle(() => resolve(got)),
            (error) => settle(() => reject(error)),
          );
        });
        const timer = setTimeout(() => {
          const why = answered
            ? "the 413 started but never finished arriving"
            : "the server waited for the body instead of refusing on Content-Length";
          settle(() => reject(new Error(why)));
        }, 2000);
        req.on("error", (error) => settle(() => reject(error)));
        req.flushHeaders();
      });
      expect(res.status).toBe(413);
      expect(res.body).toContain(`Payload Too Large: Request body must not exceed ${limit} bytes`);
      expect((await raw(port, { headers: mcpHeaders, body: initialize })).status).toBe(200);
    });

    it("refuses an oversized chunked body, and keeps serving after", async () => {
      // Exactly one byte over the limit goes out with no length, and the upload is left open, so
      // the server has to count its way to the refusal and there is nothing more to write into
      // the socket it closes after. Writing on past the limit races that close, and a client slow
      // to see the status can lose the 413 to a reset.
      const body = padded(limit + 1 - padded(0).length);
      expect(Buffer.byteLength(body)).toBe(limit + 1);
      const status = await new Promise<number>((resolve, reject) => {
        const req = request({ host: "127.0.0.1", port, method: "POST", path: "/mcp", headers: mcpHeaders }, (res) => {
          clearTimeout(timer);
          resolve(res.statusCode ?? 0);
          res.resume();
          req.destroy();
        });
        const timer = setTimeout(() => {
          req.destroy();
          reject(new Error("the server waited for more body instead of refusing at the limit"));
        }, 2000);
        req.on("error", (error) => {
          clearTimeout(timer);
          reject(error);
        });
        for (const chunk of inChunks(body)) req.write(chunk);
      });
      expect(status).toBe(413);
      expect((await raw(port, { headers: mcpHeaders, body: initialize })).status).toBe(200);
    });
  });

  it("serves only its path, and only POST", async () => {
    expect((await raw(port, { path: "/other", headers: mcpHeaders, body: initialize })).status).toBe(404);
    expect((await raw(port, { path: "//", headers: mcpHeaders, body: initialize })).status).toBe(404);
    expect((await raw(port, { path: "/mcp?x=1", headers: mcpHeaders, body: initialize })).status).toBe(200);
    const get = await raw(port, { method: "GET", headers: { accept: "text/event-stream" } });
    expect(get.status).toBe(405);
    expect(get.headers.allow).toBe("POST");
  });
});
