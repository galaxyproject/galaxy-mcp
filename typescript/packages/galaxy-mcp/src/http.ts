/**
 * The Streamable HTTP transport, with the Python server's guards.
 *
 * Over stdio the caller is whoever launched the process. Over HTTP it is anything that can
 * reach the socket -- other hosts on the network, and any web page open in a browser on the
 * same machine. This file closes the part of that gap the server can close on its own, the
 * same way and under the same settings as `python/src/galaxy_mcp/http_security.py`: which
 * Host and Origin headers it answers, and whether it will start on a routable address at all.
 * There is no OAuth on this side yet, so the unauthenticated rules are the only rules.
 *
 * Each POST gets its own server and transport (the SDK's stateless mode), so nothing is held
 * between requests but the Galaxy context the caller hands in.
 */
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { BlockList, isIP } from "node:net";
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import { ALLOW_LOCAL_FILES_ENV } from "./server.js";

export const DEFAULT_HTTP_HOST = "127.0.0.1";
export const DEFAULT_HTTP_PORT = 8000;
export const DEFAULT_HTTP_PATH = "/";

export const ALLOW_UNAUTHENTICATED_ENV = "GALAXY_MCP_ALLOW_UNAUTHENTICATED";
export const ALLOWED_HOSTS_ENV = "GALAXY_MCP_ALLOWED_HOSTS";
export const ALLOWED_ORIGINS_ENV = "GALAXY_MCP_ALLOWED_ORIGINS";

/** The requested HTTP configuration is not safe to serve. */
export class HttpStartupError extends Error {}

const loopback = new BlockList();
loopback.addSubnet("127.0.0.0", 8, "ipv4");
loopback.addAddress("::1", "ipv6");

export function isLoopbackHost(host: string): boolean {
  const candidate = host.trim().replace(/^\[|\]$/g, "").replace(/\.$/, "").toLowerCase();
  if (candidate === "localhost") return true;
  const family = isIP(candidate);
  return family !== 0 && loopback.check(candidate, family === 4 ? "ipv4" : "ipv6");
}

/** A comma-separated setting as a lower-cased set, the way the Python server reads it. */
function envList(value: string | undefined): Set<string> {
  return new Set((value ?? "").split(",").map((item) => item.trim().toLowerCase()).filter(Boolean));
}

/** A yes/no setting, read the way the Python server reads its flags. */
export function envFlag(value: string | undefined): boolean {
  return ["1", "true", "yes", "on"].includes((value ?? "").trim().toLowerCase());
}

export function checkHttpStartup(host: string, allowUnauthenticated: boolean): void {
  if (allowUnauthenticated || isLoopbackHost(host)) return;
  throw new HttpStartupError(
    `Refusing to serve HTTP on '${host}' without authentication: anyone who can reach that ` +
      "address could use every tool, with the Galaxy API key configured in this environment. " +
      `Bind to 127.0.0.1, or pass --allow-unauthenticated (${ALLOW_UNAUTHENTICATED_ENV}=1) if ` +
      "something else controls access to the listener. Clients that reach it by a non-loopback " +
      `name also need ${ALLOWED_HOSTS_ENV}.`,
  );
}

/** Command-line flags; each one wins over its environment variable. */
export interface HttpFlags {
  host?: string;
  port?: string;
  path?: string;
  allowUnauthenticated?: boolean;
}

export interface HttpConfig extends HttpGuards {
  host: string;
  port: number;
  path: string;
  allowLocalFiles: boolean;
}

/**
 * Resolves flags and environment into a configuration, with the Python server's names and
 * defaults, or throws HttpStartupError for one that should not be served.
 */
export function httpConfig(flags: HttpFlags, env: NodeJS.ProcessEnv): HttpConfig {
  const host = flags.host ?? env.GALAXY_MCP_HOST ?? DEFAULT_HTTP_HOST;
  const rawPort = flags.port ?? env.GALAXY_MCP_PORT;
  const port = rawPort === undefined ? DEFAULT_HTTP_PORT : Number(rawPort);
  // Number("") is 0, which would quietly bind a random port; Python refuses a blank one too.
  if (rawPort?.trim() === "" || !Number.isInteger(port) || port < 0 || port > 65535) {
    throw new HttpStartupError(`invalid port: ${rawPort}`);
  }
  const path = flags.path ?? env.GALAXY_MCP_HTTP_PATH ?? DEFAULT_HTTP_PATH;
  checkHttpStartup(host, flags.allowUnauthenticated === true || envFlag(env[ALLOW_UNAUTHENTICATED_ENV]));
  return {
    host,
    port,
    path: path.startsWith("/") ? path : `/${path}`,
    allowedHosts: envList(env[ALLOWED_HOSTS_ENV]),
    allowedOrigins: new Set([...envList(env[ALLOWED_ORIGINS_ENV])].map((origin) => origin.replace(/\/+$/, ""))),
    allowLocalFiles: envFlag(env[ALLOW_LOCAL_FILES_ENV]),
  };
}

function hostname(url: string): string {
  try {
    return new URL(url).hostname.replace(/^\[|\]$/g, "").toLowerCase();
  } catch {
    return "";
  }
}

export interface HttpGuards {
  allowedHosts: Set<string>;
  allowedOrigins: Set<string>;
}

/** A loopback Host, or one the operator named; defeats DNS rebinding. */
export function hostAllowed(hostHeader: string, { allowedHosts }: HttpGuards): boolean {
  if (allowedHosts.has("*")) return true;
  const name = hostname(`http://${hostHeader}`);
  return isLoopbackHost(name) || allowedHosts.has(name);
}

/** Pages served from this machine (MCP Inspector and friends) are the user's own. */
export function originAllowed(origin: string, { allowedOrigins }: HttpGuards): boolean {
  const normalized = origin.toLowerCase().replace(/\/+$/, "");
  if (allowedOrigins.has("*") || allowedOrigins.has(normalized)) return true;
  return isLoopbackHost(hostname(normalized));
}

function plain(res: ServerResponse, status: number, text: string, headers: Record<string, string> = {}): void {
  res.writeHead(status, { "content-type": "text/plain; charset=utf-8", ...headers }).end(text);
}

export interface HttpServerOptions extends HttpGuards {
  path: string;
  /** A fresh MCP server for one request. */
  newServer: () => McpServer;
}

export function createHttpHandler(options: HttpServerOptions) {
  const guards: HttpGuards = options;
  return async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    if (!hostAllowed(req.headers.host ?? "", guards)) return plain(res, 421, "Invalid Host header");

    const origin = req.headers.origin;
    const cors: Record<string, string> = {};
    if (origin !== undefined) {
      if (!originAllowed(origin, guards)) return plain(res, 403, "Origin not allowed");
      Object.assign(cors, {
        "access-control-allow-origin": origin,
        "access-control-allow-methods": req.headers["access-control-request-method"] ?? "POST,GET,OPTIONS",
        "access-control-allow-headers":
          req.headers["access-control-request-headers"] ?? "authorization,content-type,mcp-protocol-version",
        "access-control-max-age": "600",
        vary: "Origin",
      });
      if (req.method === "OPTIONS") return void res.writeHead(204, cors).end();
      for (const [name, value] of Object.entries(cors)) res.setHeader(name, value);
    }

    // The request target's path, without parsing it as a URL: a target like `//` is not one.
    if ((req.url ?? "/").split("?", 1)[0] !== options.path) return plain(res, 404, "Not Found");
    // Stateless: no session to resume a stream on (GET) or to end (DELETE).
    if (req.method !== "POST") return plain(res, 405, "Method Not Allowed", { allow: "POST" });

    const server = options.newServer();
    const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
    res.on("close", () => {
      void transport.close();
      void server.close();
    });
    try {
      await server.connect(transport);
      await transport.handleRequest(req, res);
    } catch (error) {
      console.error("galaxy-mcp: HTTP request failed:", error);
      if (!res.headersSent) plain(res, 500, "Internal Server Error");
    }
  };
}

export function startHttpServer(options: HttpServerOptions & { host: string; port: number }): Promise<Server> {
  const httpServer = createServer(createHttpHandler(options));
  return new Promise((resolve, reject) => {
    httpServer.once("error", reject);
    httpServer.listen(options.port, options.host, () => {
      // Past startup a server error would reach a promise that has already settled and vanish.
      httpServer.off("error", reject);
      resolve(httpServer);
    });
  });
}
