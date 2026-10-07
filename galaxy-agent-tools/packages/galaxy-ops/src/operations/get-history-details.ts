import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { httpError } from "../errors";
import { pyGet, pyStr } from "../python-values";
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
  if (error || !data) throw httpError(response, error);
  const history = data as HistoryDetail;
  const contents = await ctx.client.GET("/api/histories/{history_id}/contents", {
    params: { path: { history_id: i.historyId } },
  });
  if (contents.error || !contents.data) {
    throw httpError(contents.response, contents.error);
  }
  recordFact(ctx, contentsCount, Array.isArray(contents.data) ? contents.data.length : 0);
  return history;
}

/**
 * The sentence the other server puts beside the number, word for word.
 *
 * It is there because a count reads like a listing to an agent that asked for a
 * history's details and got one number back, so the payload says which tool
 * returns the datasets themselves.
 */
const CONTENTS_NOTE =
  "This is just a count. To get actual datasets, use " +
  "get_history_contents(history_id, limit=25, order='create_time-dsc') " +
  "for newest datasets first.";

export const getHistoryDetailsOp: Operation<typeof input, HistoryDetail> = {
  name: "get_history_details",
  domain: "histories",
  result: { kind: "object", fields: ["history", "contents_summary"] },
  summary: "Show a single history's details by id (name, state, counts).",
  input,
  run,
  // The other server wraps the record: this tool answers with the history under its
  // own key and the count it already paid a request for under another, so a caller
  // reading `data` cannot mistake one number for the history's own metadata. run()
  // goes on returning the bare record, which is what a library caller destructures.
  project: (h, i, facts) => {
    const total = readFact(facts, contentsCount) ?? 0;
    return {
      data: { history: h, contents_summary: { total_items: total, note: CONTENTS_NOTE } },
      // server.py, get_history_details: the record's name through dict.get, so a
      // history with no name at all is named by the id that was asked for.
      message: `Retrieved details for history '${pyStr(pyGet(h as Record<string, unknown>, "name", i.historyId))}'`,
      count: total,
    };
  },
  // server.py, get_history_details: a 404 keeps the tool's own sentence, which says which id
  // was not found and that this argument is a string rather than a history object -- passing
  // the repr of one is the mistake that brings people here. Anything else goes through
  // format_error.
  failure: {
    shape: "bioblend-get",
    action: "Get history details",
    context: (i) => ({ history_id: i.historyId }),
    sentence: (_text, status, i) =>
      status === 404
        ? `History ID '${i.historyId}' not found. Make sure to pass a valid history ID string.`
        : undefined,
  },
};

register(getHistoryDetailsOp as AnyOperation);

export const getHistoryDetails = (i: In, ctx: GalaxyContext) => runOperation(getHistoryDetailsOp, i, ctx);
