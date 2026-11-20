import { normalizeBaseUrl } from "../base-url";
import { AuthError, type AuthDeps, type BaseUrl } from "../types";
import { fail, hostOf, requireLoginAt } from "./common";

export interface HeadersOptions {
  host?: string;
}

/**
 * The Galaxy that Claude Code's MCP server URL names: its origin and path prefix, minus the
 * `/api/mcp` endpoint. Claude Code sets `CLAUDE_CODE_MCP_SERVER_URL` for `headersHelper`
 * commands, so the helper can tell which server it is being asked to authenticate to.
 *
 * A URL that will not parse is a refusal (77, like every other refusal to print a token), not a
 * guess at a host.
 */
function galaxyOfMcpUrl(serverUrl: string): BaseUrl {
  try {
    const url = new URL(serverUrl);
    url.search = "";
    url.hash = "";
    url.pathname = url.pathname.replace(/\/+$/, "").replace(/\/api\/mcp$/, "");
    return normalizeBaseUrl(url.toString());
  } catch (e) {
    throw new AuthError(
      "not_logged_in",
      `Refusing to print a token: cannot tell which Galaxy CLAUDE_CODE_MCP_SERVER_URL names (${e instanceof Error ? e.message : String(e)}).`,
    );
  }
}

/**
 * For Claude Code's MCP `headersHelper`: a JSON object of headers on stdout. A helper must
 * never hand a Galaxy token to a server that is not that Galaxy, so when Claude Code names
 * the server, only the stored login for exactly that Galaxy qualifies. The server URL comes
 * from the MCP config, and a project's `.mcp.json` can name any host: without this check a
 * repository could point the server at its own address and have the token printed for it.
 *
 * The header is `x-api-key`, the one galaxy-ops sends for every operation. The MCP authorization
 * spec has clients send `Authorization: Bearer`, which Galaxy's API reads only when no key is
 * given; this does not send it.
 *
 * With no server named it refuses, rather than pick a host by the ordinary rule: that would hand
 * the only stored login's token to whatever server asked, which is the one thing this check is
 * for. `--host` alone is not enough either, since a project's `.mcp.json` can set the helper
 * command as well as the URL. To print a token by hand, `gxy auth token` is the command.
 */
export async function headersCommand(opts: HeadersOptions, deps: AuthDeps): Promise<number> {
  try {
    const serverUrl = deps.env.CLAUDE_CODE_MCP_SERVER_URL;
    // `=== undefined` only: a server that is named but blank is still named, and is refused by
    // `galaxyOfMcpUrl`.
    if (serverUrl === undefined) {
      throw new AuthError(
        "not_logged_in",
        `Refusing to print a token: CLAUDE_CODE_MCP_SERVER_URL is not set, so there is no server to check it against. ` +
          `\`${deps.commandName} auth headers\` is for an MCP headersHelper, which Claude Code runs with it set; ` +
          `\`${deps.commandName} auth token\` prints a token.`,
      );
    }
    const target = galaxyOfMcpUrl(serverUrl);
    // `--host` may agree with the server Claude Code named but never move the token to another.
    if (opts.host) {
      const named = await deps.resolveHost(opts.host);
      if (named !== target) {
        throw new AuthError(
          "not_logged_in",
          `Refusing to print a token: --host ${hostOf(named)} is not the server Claude Code is connecting to (${hostOf(target)}).`,
        );
      }
    }
    const login = await requireLoginAt(deps, target);
    deps.io.stdout.write(`${JSON.stringify({ "x-api-key": login.token })}\n`);
    return 0;
  } catch (e) {
    return fail(deps, e);
  }
}
