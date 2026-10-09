import type { GetJson } from "../bindings";
import type { GalaxyContext, GalaxyVersionSource } from "../context";
import { httpError } from "../errors";
import { satisfiesRequirement } from "../version";
import { allOperations, register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** A tool this server is too old to run, and the bound it misses. */
export interface UnsupportedTool {
  name: string;
  requires: string;
}

/**
 * The sixteen configuration fields this tool reports, and only those.
 *
 * Galaxy's /api/configuration is long and grows; the other server lifts these out by name
 * and answers with nothing else, so a client reading `data.config` reads the same set from
 * either. Each is typed `unknown` because that is what the endpoint promises for most of
 * them and what a reader has to narrow anyway.
 */
export interface ServerConfigSummary {
  brand: unknown;
  logo_url: unknown;
  welcome_url: unknown;
  support_url: unknown;
  citation_url: unknown;
  terms_url: unknown;
  allow_user_creation: unknown;
  allow_user_deletion: unknown;
  enable_quotas: unknown;
  ftp_upload_site: unknown;
  wiki_url: unknown;
  screencasts_url: unknown;
  library_import_dir: unknown;
  user_library_import_dir: unknown;
  allow_library_path_paste: unknown;
  enable_unique_workflow_defaults: unknown;
}

export interface ServerInfo {
  url: string;
  version: GetJson<"/api/version">;
  config: ServerConfigSummary;
  /**
   * Whether version_major could be read at all. When false, unsupported_tools is empty because
   * nothing is known, not because everything is supported -- and nothing will be refused.
   */
  version_known: boolean;
  /** The tools this server cannot run. Empty on a new enough server and on an unreadable one. */
  unsupported_tools: UnsupportedTool[];
  /**
   * Where `version` came from: "server" when it was fetched, "supplied" when the caller
   * handed the context a `serverVersion` and nothing was asked of the server, "unknown" when
   * neither. A library caller is the only one who can supply a version, so this is the
   * library result's to say -- it stays off the wire, where the other server has no such key
   * and a caller cannot have supplied one.
   */
  version_source: GalaxyVersionSource;
}

/**
 * Every field the summary carries, and what a configuration that does not mention it reads as.
 *
 * `dict.get(key, default)` substitutes for an ABSENT key and not for a null Galaxy sent, so
 * these are applied by asking whether the key is there rather than by coalescing -- a brand
 * Galaxy sent as null stays null and does not become "Galaxy".
 */
const CONFIG_FIELDS: ReadonlyArray<[keyof ServerConfigSummary, unknown]> = [
  ["brand", "Galaxy"],
  ["logo_url", null],
  ["welcome_url", null],
  ["support_url", null],
  ["citation_url", null],
  ["terms_url", null],
  ["allow_user_creation", null],
  ["allow_user_deletion", null],
  ["enable_quotas", null],
  ["ftp_upload_site", null],
  ["wiki_url", null],
  ["screencasts_url", null],
  ["library_import_dir", null],
  ["user_library_import_dir", null],
  ["allow_library_path_paste", null],
  ["enable_unique_workflow_defaults", null],
];

function summarizeConfig(config: unknown): ServerConfigSummary {
  // Own keys only, read off the parsed body: a configuration that happens to arrive with a
  // null prototype has no inherited `constructor` to mistake for a field, and one that does
  // must not have `toString` read as its brand.
  const sent = (config ?? {}) as Record<string, unknown>;
  const has = (key: string) => Object.prototype.hasOwnProperty.call(sent, key);
  return Object.fromEntries(
    CONFIG_FIELDS.map(([key, fallback]) => [key, has(key) ? sent[key] : fallback]),
  ) as unknown as ServerConfigSummary;
}

const input = {}; // no args

async function run(_in: Record<string, never>, ctx: GalaxyContext): Promise<ServerInfo> {
  // Through the context's lookup rather than a probe of its own. Two probes can reach two
  // answers -- one good response cached here and a later 401 seen by the guard -- and then this
  // op reports a set of refusals that will not happen. One lookup, one answer, and the answer
  // this op obtains is the one every later guard sees.
  const { version, payload, error, source = "unknown" } = (await ctx.galaxyVersion?.()) ?? {};
  if (error) throw error;
  const c = await ctx.client.GET("/api/configuration", {});
  if (c.error || !c.data) throw httpError(c.response, c.error);
  // Sorted by name, because the other server walks a dict of requirements through
  // `sorted()` and a list in registration order would be the same set in a different order.
  const unsupported: UnsupportedTool[] = version
    ? allOperations
        .flatMap((op) =>
          op.requires && !satisfiesRequirement(version, op.requires.galaxy)
            ? [{ name: op.name, requires: op.requires.galaxy }]
            : [],
        )
        .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
    : [];
  const info: ServerInfo = {
    url: ctx.baseUrl ?? "",
    // A supplied version was never fetched, so there is no payload to hand back -- but it is
    // the version being enforced, and answering "version ?" for one we know is worse than
    // answering with the one field we can honestly fill.
    version: (payload ??
      (version ? { version_major: `${version.major}.${version.minor}` } : {})) as ServerInfo["version"],
    config: summarizeConfig(c.data),
    version_known: version !== undefined,
    unsupported_tools: unsupported,
    version_source: source,
  };
  return info;
}

export const getServerInfoOp: Operation<typeof input, ServerInfo> = {
  name: "get_server_info",
  domain: "connection",
  summary:
    "Return the connected Galaxy's URL, version, and the sixteen public configuration fields " +
    "the other server reports, plus `unsupported_tools` -- the tools this server is too old " +
    "to run.",
  input,
  run,
  // server.py, get_server_info: the address that was connected to, and nothing else.
  // The version, whether it could be read, and the tools this server is too old for are
  // all in data, which is where a caller reads them. `version_source` is not: the other
  // server has no such key, and no wire caller can supply a version, so the fact stays on
  // the library result and comes off here -- the way trimmedForSize does.
  project: (s) => {
    const { version_source: _source, ...data } = s;
    return { data, message: `Retrieved server info for ${s.url}` };
  },
  // server.py, get_server_info: its own sentence. The configuration is read first there,
  // and the version after it, so a server that answers one and not the other fails on the
  // same request on both sides.
  failure: {
    shape: "bioblend-get",
    sentence: (text) => `Failed to get server information: ${text}`,
  },
};

register(getServerInfoOp as AnyOperation);

export const getServerInfo = (i: Record<string, never>, ctx: GalaxyContext) => runOperation(getServerInfoOp, i, ctx);
