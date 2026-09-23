// Note: /api/histories/{history_id}/contents/{dataset_id}/provenance is not in the typed API paths.
// We route it through legacyGet, which handles the any-cast internally and throws typed errors.
import { z } from "zod";
import type { GalaxyContext } from "../context";
import { classifyHttp, GalaxyNotFoundError } from "../errors";
import { legacyGet } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

// The provenance endpoint's job_id field is not always typed; hand-type for safety.
interface ProvenanceResponse {
  job_id?: string;
  [key: string]: unknown;
}

// Similarly, the dataset endpoint's creating_job field.
interface DatasetMeta {
  creating_job?: string;
  [key: string]: unknown;
}

export interface JobDetail {
  id?: string;
  state?: string;
  [key: string]: unknown;
}

export interface GetJobDetailsResult {
  job: JobDetail;
  dataset_id: string;
  job_id: string;
}

const input = {
  datasetId: z.string().describe("dataset (HDA) id"),
  historyId: z.string().optional().describe("history id; speeds provenance lookup"),
};
type In = { datasetId: string; historyId?: string };

async function run(i: In, ctx: GalaxyContext): Promise<GetJobDetailsResult> {
  let jobId: string | undefined;

  // Try provenance path if historyId supplied (off-schema; routed through legacyGet).
  // 404 means the provenance record doesn't exist -- fall through to dataset lookup.
  // Any other error (401/403/5xx) is a real problem and must surface to the caller.
  if (i.historyId) {
    try {
      const prov = await legacyGet<ProvenanceResponse>(ctx, "/api/histories/{history_id}/contents/{dataset_id}/provenance", {
        params: { path: { history_id: i.historyId, dataset_id: i.datasetId } },
      });
      jobId = prov.job_id;
    } catch (err) {
      if (!(err instanceof GalaxyNotFoundError)) {
        throw err;
      }
      // 404: provenance not available, fall through to dataset creating_job path
    }
    // Also fall through if provenance returned 200 but had no job_id
  }

  // Fallback: read creating_job from dataset metadata
  if (!jobId) {
    const { data, error, response } = await ctx.client.GET("/api/datasets/{dataset_id}", {
      params: { path: { dataset_id: i.datasetId } },
    });
    if (error || !data) throw classifyHttp(response.status, error);
    jobId = (data as DatasetMeta).creating_job;
    if (!jobId) throw classifyHttp(404, { err_msg: `No job found for dataset ${i.datasetId}` });
  }

  const { data: jobData, error: jobError, response: jobResp } = await ctx.client.GET("/api/jobs/{job_id}", {
    params: { path: { job_id: jobId } },
  });
  if (jobError || !jobData) throw classifyHttp(jobResp.status, jobError);

  return {
    job: jobData as JobDetail,
    dataset_id: i.datasetId,
    job_id: jobId,
  };
}

export const getJobDetailsOp: Operation<typeof input, GetJobDetailsResult> = {
  name: "get_job_details",
  domain: "jobs",
  summary: "Get job details for the job that produced a dataset.",
  input,
  run,
  project: (out) => ({ message: `Job ${out.job_id} for dataset ${out.dataset_id}` }),
};

register(getJobDetailsOp as AnyOperation);

export const getJobDetails = (i: In, ctx: GalaxyContext) => runOperation(getJobDetailsOp, i, ctx);
