import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { request, type Server } from "node:http";
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

/** A raw request, so the test controls the Host header that fetch would not let it set. */
function raw(
  port: number,
  { method = "POST", path = "/mcp", headers = {}, body }: { method?: string; path?: string; headers?: Record<string, string>; body?: string },
): Promise<{ status: number; headers: Record<string, string | string[] | undefined>; body: string }> {
  return new Promise((resolve, reject) => {
    const req = request({ host: "127.0.0.1", port, method, path, headers }, (res) => {
      let text = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => (text += chunk));
      res.on("end", () => resolve({ status: res.statusCode ?? 0, headers: res.headers, body: text }));
    });
    req.on("error", reject);
    req.end(body);
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

  it("serves only its path, and only POST", async () => {
    expect((await raw(port, { path: "/other", headers: mcpHeaders, body: initialize })).status).toBe(404);
    expect((await raw(port, { path: "//", headers: mcpHeaders, body: initialize })).status).toBe(404);
    expect((await raw(port, { path: "/mcp?x=1", headers: mcpHeaders, body: initialize })).status).toBe(200);
    const get = await raw(port, { method: "GET", headers: { accept: "text/event-stream" } });
    expect(get.status).toBe(405);
    expect(get.headers.allow).toBe("POST");
  });
});
