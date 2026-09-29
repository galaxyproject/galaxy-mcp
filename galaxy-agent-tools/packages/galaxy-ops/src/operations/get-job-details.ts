// Note: /api/histories/{history_id}/contents/{dataset_id}/provenance is not in the typed API paths.
// We route it through legacyGet, which handles the any-cast internally and throws typed errors.
import { z } from "zod";
import type { GalaxyContext } from "../context";
import { httpError, GalaxyNotFoundError } from "../errors";
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
  historyId: z.string().nullish().describe("history id; speeds provenance lookup"),
};
type In = { datasetId: string; historyId?: string };

async function run(i: In, ctx: GalaxyContext): Promise<GetJobDetailsResult> {
  let jobId: string | undefined;
  // What provenance said, kept rather than thrown. The other server holds this failure back
  // and only reports it if the fallback ALSO fails or finds no job: a dataset whose
  // provenance record is out of reach still names its creating job, and answering with the
  // provenance failure there would refuse a question that had an answer.
  let provenanceError: unknown;

  if (i.historyId) {
    try {
      const prov = await legacyGet<ProvenanceResponse>(ctx, "/api/histories/{history_id}/contents/{dataset_id}/provenance", {
        params: { path: { history_id: i.historyId, dataset_id: i.datasetId } },
      });
      jobId = prov.job_id;
    } catch (err) {
      provenanceError = err;
    }
    // Also fall through if provenance returned 200 but had no job_id
  }

  // Fallback: read creating_job from dataset metadata
  if (!jobId) {
    const { data, error, response } = await ctx.client.GET("/api/datasets/{dataset_id}", {
      params: { path: { dataset_id: i.datasetId } },
    });
    if (error || !data) throw provenanceError ?? httpError(response, error);
    jobId = (data as DatasetMeta).creating_job;
    if (!jobId) {
      if (provenanceError) throw provenanceError;
      throw new GalaxyNotFoundError(
        `No job information found for dataset '${i.datasetId}'. ` +
          "The dataset may not have been created by a job.",
      );
    }
  }

  const { data: jobData, error: jobError, response: jobResp } = await ctx.client.GET("/api/jobs/{job_id}", {
    params: { path: { job_id: jobId } },
  });
  if (jobError || !jobData) throw httpError(jobResp, jobError);

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
  // server.py, get_job_details: the dataset that was asked about, not the job that
  // was found for it -- the job's id is in `data`.
  project: (out) => ({ message: `Retrieved job details for dataset '${out.dataset_id}'` }),
  // server.py, _job_details_failed: every failure here is described from the exception that
  // actually failed, and a 404 keeps the tool's own sentence because a 404 from the jobs API
  // is as likely to be a permission problem as a missing dataset. The two routes are asked
  // through two different clients over there -- bioblend for the dataset and provenance,
  // requests itself for the job -- so they do not word a failure the same way.
  failure: {
    shape: (facts) => (facts.url.includes("/api/jobs/") ? "raise-for-status" : "bioblend-get"),
    action: "Get job details",
    context: (i) => ({ dataset_id: i.datasetId }),
    sentence: (_text, status, i) =>
      status === 404
        ? `Dataset ID '${i.datasetId}' not found or job not accessible. ` +
          "Make sure the dataset exists and you have permission to view it."
        : undefined,
  },
};

register(getJobDetailsOp as AnyOperation);

export const getJobDetails = (i: In, ctx: GalaxyContext) => runOperation(getJobDetailsOp, i, ctx);
