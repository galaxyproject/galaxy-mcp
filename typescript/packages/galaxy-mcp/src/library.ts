/**
 * The server as a library: for a program that builds the MCP server itself, for example to
 * serve it behind its own authentication, rather than running the `galaxy-mcp` command.
 */
export { buildServer, toolNames, toolAnnotations, ALLOW_LOCAL_FILES_ENV, APPS_ENV, type ServerOptions } from "./server.js";
export {
  createHttpHandler,
  startHttpServer,
  httpConfig,
  HttpStartupError,
  type HttpConfig,
  type HttpFlags,
  type HttpGuards,
  type HttpServerOptions,
} from "./http.js";
