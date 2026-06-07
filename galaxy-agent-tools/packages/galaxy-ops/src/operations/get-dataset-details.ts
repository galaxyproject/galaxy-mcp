import { z } from "zod";
import type { GetJson } from "../bindings";
import type { GalaxyContext } from "../context";
import { classifyHttp } from "../errors";
import { register } from "./registry";
import type { AnyOperation, Operation } from "./types";

export type DatasetDetail = GetJson<"/api/datasets/{dataset_id}">;

const input = { datasetId: z.string().describe("Encoded dataset id") };
type In = { datasetId: string };

async function run(i: In, ctx: GalaxyContext): Promise<DatasetDetail> {
  const { data, error, response } = await ctx.client.GET("/api/datasets/{dataset_id}", {
    params: { path: { dataset_id: i.datasetId } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  return data as DatasetDetail;
}

export const getDatasetDetailsOp: Operation<typeof input, DatasetDetail> = {
  name: "get_dataset_details",
  domain: "datasets",
  summary: "Show a dataset's metadata by id (state, extension, name). Content preview lands in a later phase.",
  input,
  run,
  project: (d) => ({ message: `Dataset ${(d as { id?: string }).id} state=${(d as { state?: string }).state}` }),
};

register(getDatasetDetailsOp as AnyOperation);

export const getDatasetDetails = (i: In, ctx: GalaxyContext) => getDatasetDetailsOp.run(i, ctx);
