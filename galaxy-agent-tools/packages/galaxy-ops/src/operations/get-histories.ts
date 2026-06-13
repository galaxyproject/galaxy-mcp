import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register } from "./registry";
import type { AnyOperation, Operation, Pagination } from "./types";

export type Histories = GetJson<"/api/histories">;

const input = {
  limit: z.coerce.number().int().positive().optional().describe("Max histories to return"),
  offset: z.coerce.number().int().min(0).optional().describe("Skip the first N"),
  name: z.string().optional().describe("Case-insensitive substring filter on history name"),
};
type In = { limit?: number; offset?: number; name?: string };

async function run(i: In, ctx: GalaxyContext): Promise<Histories> {
  const { data, error, response } = await ctx.client.GET("/api/histories", {
    params: { query: { limit: i.limit ?? null, offset: i.offset ?? null } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  if (!i.name) return data;
  const needle = i.name.toLowerCase();
  return (data as Array<{ name?: string }>).filter((h) => (h.name ?? "").toLowerCase().includes(needle)) as Histories;
}

export const getHistoriesOp: Operation<typeof input, Histories> = {
  name: "get_histories",
  domain: "histories",
  summary: "List the current user's histories (id, name, counts). Optional name substring filter.",
  input,
  run,
  project: (hs) => {
    const arr = hs as unknown[];
    const pagination: Pagination = { total: arr.length };
    return { message: `${arr.length} histor${arr.length === 1 ? "y" : "ies"}`, pagination };
  },
};

register(getHistoriesOp as AnyOperation);

export const getHistories = (i: In, ctx: GalaxyContext) => getHistoriesOp.run(i, ctx);
