import type { GalaxyContext } from "../context";
import { register } from "./registry";
import { getHistoriesOp } from "./get-histories";
import type { AnyOperation, Operation } from "./types";

export interface HistoryRef { id: string; name: string; }

const input = {};

async function run(_in: Record<string, never>, ctx: GalaxyContext): Promise<HistoryRef[]> {
  const histories = (await getHistoriesOp.run({}, ctx)) as Array<{ id?: string; name?: string }>;
  return histories.map((h) => ({ id: h.id ?? "", name: h.name ?? "" }));
}

export const listHistoryIdsOp: Operation<typeof input, HistoryRef[]> = {
  name: "list_history_ids",
  domain: "histories",
  summary: "List just the id and name of each history (compact picker for agents).",
  input,
  run,
  project: (rows) => ({ message: `${rows.length} histor${rows.length === 1 ? "y" : "ies"}` }),
};

register(listHistoryIdsOp as AnyOperation);

export const listHistoryIds = (i: Record<string, never>, ctx: GalaxyContext) => listHistoryIdsOp.run(i, ctx);
