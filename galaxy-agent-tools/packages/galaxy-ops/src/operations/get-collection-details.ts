import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type CollectionDetail = GetJson<"/api/dataset_collections/{hdca_id}">;

const input = {
  collectionId: z.string().describe("Encoded HDCA (history dataset collection) id"),
  maxElements: z.coerce.number().int().positive().optional().describe("Truncate the elements list to N"),
};
type In = { collectionId: string; maxElements?: number };

async function run(i: In, ctx: GalaxyContext): Promise<CollectionDetail> {
  const { data, error, response } = await ctx.client.GET("/api/dataset_collections/{hdca_id}", {
    params: { path: { hdca_id: i.collectionId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  if (i.maxElements != null) {
    const d = data as { elements?: unknown[] };
    if (Array.isArray(d.elements)) d.elements = d.elements.slice(0, i.maxElements);
  }
  return data as CollectionDetail;
}

export const getCollectionDetailsOp: Operation<typeof input, CollectionDetail> = {
  name: "get_collection_details",
  domain: "collections",
  summary: "Show a dataset collection by id, with its elements (optionally truncated).",
  input,
  run,
  project: (c) => ({ message: `Collection ${(c as { id?: string }).id} (${((c as { elements?: unknown[] }).elements ?? []).length} elements)` }),
};

register(getCollectionDetailsOp as AnyOperation);

export const getCollectionDetails = (i: In, ctx: GalaxyContext) => getCollectionDetailsOp.run(i, ctx);
