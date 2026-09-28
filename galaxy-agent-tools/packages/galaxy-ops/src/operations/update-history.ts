import { z } from "zod";
import type { PutJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { httpError, GalaxyValidationError } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type UpdatedHistory = PutJson<"/api/histories/{history_id}">;

const input = {
  historyId: z.string().describe("History id to update"),
  name: z.string().optional().describe("New name"),
  annotation: z.string().optional().describe("New annotation"),
  tags: z.array(z.string()).optional().describe("New tags (replaces existing)"),
  deleted: z.boolean().optional().describe("Soft-delete or restore"),
  published: z.boolean().optional().describe("Publish or unpublish"),
};

type In = {
  historyId: string;
  name?: string;
  annotation?: string;
  tags?: string[];
  deleted?: boolean;
  published?: boolean;
};

async function run(i: In, ctx: GalaxyContext): Promise<UpdatedHistory> {
  const body: Record<string, unknown> = {};
  if (i.name !== undefined) body["name"] = i.name;
  if (i.annotation !== undefined) body["annotation"] = i.annotation;
  if (i.tags !== undefined) body["tags"] = i.tags;
  if (i.deleted !== undefined) body["deleted"] = i.deleted;
  if (i.published !== undefined) body["published"] = i.published;

  if (Object.keys(body).length === 0) {
    // server.py, update_history: refused before the connection is even checked, so the
    // sentence is the whole answer and names the fields it would have taken.
    throw new GalaxyValidationError(
      "No fields provided to update. Pass at least one of: " +
        "name, annotation, tags, deleted, published.",
    );
  }

  const { data, error, response } = await ctx.client.PUT("/api/histories/{history_id}", {
    params: { path: { history_id: i.historyId } },
    body: body as never,
  });
  if (error || !data) throw httpError(response, error);
  return data as UpdatedHistory;
}

export const updateHistoryOp: Operation<typeof input, UpdatedHistory> = {
  name: "update_history",
  domain: "histories",
  summary: "Update history metadata (name, annotation, tags, deleted, published).",
  input,
  readOnly: false,
  run,
  project: (_data, i) => {
    const changed = (["name", "annotation", "tags", "deleted", "published"] as const).filter(
      (k) => i[k] !== undefined,
    );
    return { message: `Updated history ${i.historyId} (${changed.join(", ")})` };
  },
  // server.py, update_history: format_error with two arguments, so nothing is appended.
  failure: { shape: "bioblend-write", action: "Update history" },
};

register(updateHistoryOp as AnyOperation);

export const updateHistory = (i: In, ctx: GalaxyContext) => runOperation(updateHistoryOp, i, ctx);
