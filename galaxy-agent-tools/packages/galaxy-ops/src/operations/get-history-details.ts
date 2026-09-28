import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { envelopeFact, readFact, recordFact } from "./envelope-facts";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type HistoryDetail = GetJson<"/api/histories/{history_id}">;

const input = { historyId: z.string().describe("Encoded history id") };
type In = { historyId: string };

/**
 * How many items the history holds, for the record run() just returned.
 *
 * The other surface answers this tool with a count of everything in the history,
 * which it gets by asking for the contents and measuring them -- the metadata
 * carries no number that means the same thing, because the contents index
 * includes the deleted and the hidden. So the second request happens here too,
 * and the number is recorded beside the call rather than returned: run() goes on
 * returning the history exactly as Galaxy sent it.
 */
const contentsCount = envelopeFact<number>("get_history_details.contents_count");

async function run(i: In, ctx: GalaxyContext): Promise<HistoryDetail> {
  const { data, error, response } = await ctx.client.GET("/api/histories/{history_id}", {
    params: { path: { history_id: i.historyId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  const history = data as HistoryDetail;
  const contents = await ctx.client.GET("/api/histories/{history_id}/contents", {
    params: { path: { history_id: i.historyId } },
  });
  if (contents.error || !contents.data) {
    throw classifyHttp(contents.response.status, contents.error);
  }
  recordFact(ctx, contentsCount, Array.isArray(contents.data) ? contents.data.length : 0);
  return history;
}

export const getHistoryDetailsOp: Operation<typeof input, HistoryDetail> = {
  name: "get_history_details",
  domain: "histories",
  summary: "Show a single history's details by id (name, state, counts).",
  input,
  run,
  project: (h, _i, facts) => ({
    message: `History ${(h as { id?: string }).id} state=${(h as { state?: string }).state}`,
    count: readFact(facts, contentsCount) ?? null,
  }),
};

register(getHistoryDetailsOp as AnyOperation);

export const getHistoryDetails = (i: In, ctx: GalaxyContext) => runOperation(getHistoryDetailsOp, i, ctx);
