import type { GetJson } from "../bindings";
import type { GalaxyContext, GalaxyVersionSource } from "../context";
import { classifyHttp } from "../errors";
import { satisfiesRequirement } from "../version";
import { allOperations, register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** An op this server is too old to run, and the bound it misses. */
export interface UnsupportedOp {
  name: string;
  requires: string;
}

export interface ServerInfo {
  url: string;
  version: GetJson<"/api/version">;
  config: GetJson<"/api/configuration">;
  /**
   * Whether version_major could be read at all. When false, unsupported_ops is empty because
   * nothing is known, not because everything is supported -- and nothing will be refused.
   */
  version_known: boolean;
  /**
   * Where the version being enforced came from. "supplied" means the caller passed
   * serverVersion and the server was never asked, so `version` carries only version_major.
   */
  version_source: GalaxyVersionSource;
  /** The ops this server cannot run. Empty on a new enough server and on an unreadable one. */
  unsupported_ops: UnsupportedOp[];
}

const input = {}; // no args

async function run(_in: Record<string, never>, ctx: GalaxyContext): Promise<ServerInfo> {
  // Through the context's lookup rather than a probe of its own. Two probes can reach two
  // answers -- one good response cached here and a later 401 seen by the guard -- and then this
  // op reports a set of refusals that will not happen. One lookup, one answer, and the answer
  // this op obtains is the one every later guard sees.
  const { version, payload, error, source } = (await ctx.galaxyVersion?.()) ?? {
    source: "unknown" as const,
  };
  if (error) throw error;
  const c = await ctx.client.GET("/api/configuration", {});
  if (c.error || !c.data) throw classifyHttp(c.response.status, c.error);
  const unsupported: UnsupportedOp[] = version
    ? allOperations.flatMap((op) =>
        op.requires && !satisfiesRequirement(version, op.requires.galaxy)
          ? [{ name: op.name, requires: op.requires.galaxy }]
          : [],
      )
    : [];
  return {
    url: ctx.baseUrl ?? "",
    // A supplied version was never fetched, so there is no payload to hand back -- but it is
    // the version being enforced, and answering "version ?" for one we know is worse than
    // answering with the one field we can honestly fill.
    version: (payload ??
      (version ? { version_major: `${version.major}.${version.minor}` } : {})) as ServerInfo["version"],
    config: c.data,
    version_known: version !== undefined,
    version_source: source,
    unsupported_ops: unsupported,
  };
}

export const getServerInfoOp: Operation<typeof input, ServerInfo> = {
  name: "get_server_info",
  domain: "connection",
  summary:
    "Return the connected Galaxy's URL, version, and public configuration, plus " +
    "`unsupported_ops` -- the operations this server is too old to run.",
  input,
  run,
  project: (s) => ({
    message:
      `Galaxy at ${s.url} (version ${(s.version as { version_major?: string }).version_major ?? "?"}` +
      `${s.version_source === "supplied" ? ", supplied" : ""})` +
      (s.unsupported_ops.length > 0 ? `, ${s.unsupported_ops.length} op(s) unsupported` : ""),
  }),
};

register(getServerInfoOp as AnyOperation);

export const getServerInfo = (i: Record<string, never>, ctx: GalaxyContext) => runOperation(getServerInfoOp, i, ctx);
