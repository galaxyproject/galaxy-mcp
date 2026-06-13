import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type HistoryDetail = GetJson<"/api/histories/{history_id}">;

const input = { historyId: z.string().describe("Encoded history id") };
type In = { historyId: string };

async function run(i: In, ctx: GalaxyContext): Promise<HistoryDetail> {
  const { data, error, response } = await ctx.client.GET("/api/histories/{history_id}", {
    params: { path: { history_id: i.historyId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  return data as HistoryDetail;
}

export const getHistoryDetailsOp: Operation<typeof input, HistoryDetail> = {
  name: "get_history_details",
  domain: "histories",
  summary: "Show a single history's details by id (name, state, counts).",
  input,
  run,
  project: (h) => ({ message: `History ${(h as { id?: string }).id} state=${(h as { state?: string }).state}` }),
};

register(getHistoryDetailsOp as AnyOperation);

export const getHistoryDetails = (i: In, ctx: GalaxyContext) => getHistoryDetailsOp.run(i, ctx);
