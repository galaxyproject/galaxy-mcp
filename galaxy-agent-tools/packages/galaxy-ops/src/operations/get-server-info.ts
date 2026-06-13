import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

export interface ServerInfo {
  url: string;
  version: GetJson<"/api/version">;
  config: GetJson<"/api/configuration">;
}

const input = {}; // no args

async function run(_in: Record<string, never>, ctx: GalaxyContext): Promise<ServerInfo> {
  const v = await ctx.client.GET("/api/version", {});
  if (v.error || !v.data) throw classifyHttp(v.response.status, v.error);
  const c = await ctx.client.GET("/api/configuration", {});
  if (c.error || !c.data) throw classifyHttp(c.response.status, c.error);
  return { url: ctx.baseUrl ?? "", version: v.data, config: c.data };
}

export const getServerInfoOp: Operation<typeof input, ServerInfo> = {
  name: "get_server_info",
  domain: "connection",
  summary: "Return the connected Galaxy's URL, version, and public configuration.",
  input,
  run,
  project: (s) => ({ message: `Galaxy at ${s.url} (version ${(s.version as { version_major?: string }).version_major ?? "?"})` }),
};

register(getServerInfoOp as AnyOperation);

export const getServerInfo = (i: Record<string, never>, ctx: GalaxyContext) => getServerInfoOp.run(i, ctx);
