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

/** The log fields Galaxy adds to a job when asked for it in full, and how much of each to keep. */
const JOB_LOG_FIELDS = ["tool_stdout", "tool_stderr", "job_stdout", "job_stderr", "stdout", "stderr"];
const JOB_LOG_BYTES = 4 * 1024;

/**
 * Both ends of a log, cut on line boundaries: the cause is usually at the end, the context at
 * the start, and a long log keeps neither if only one end is read. Measured in UTF-8 bytes, with
 * a line saying how much was left out -- the other server's `_ends`, byte for byte.
 */
export function logEnds(text: string, cap = JOB_LOG_BYTES): string {
  const data = new TextEncoder().encode(text);
  if (data.length <= cap) return text;
  const half = Math.floor(cap / 2);
  const front = data.subarray(0, half);
  const back = data.subarray(data.length - half);
  const cut = front.lastIndexOf(10);
  const head = cut < 0 ? front : front.subarray(0, cut);
  const start = back.indexOf(10);
  const tail = start < 0 ? back : back.subarray(start + 1);
  const dropped = data.length - head.length - tail.length;
  const decode = (bytes: Uint8Array) => new TextDecoder().decode(bytes);
  return `${decode(head)}\n[... ${dropped} of ${data.length} bytes omitted ...]\n${decode(tail)}`;
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

  // In full: a failed job's logs are only in the full view.
  const { data: jobData, error: jobError, response: jobResp } = await ctx.client.GET("/api/jobs/{job_id}", {
    params: { path: { job_id: jobId }, query: { full: true } },
  });
  if (jobError || !jobData) throw httpError(jobResp, jobError);
  const job = { ...(jobData as JobDetail) };
  for (const field of JOB_LOG_FIELDS) {
    if (typeof job[field] === "string") job[field] = logEnds(job[field] as string);
  }

  return {
    job,
    dataset_id: i.datasetId,
    job_id: jobId,
  };
}

export const getJobDetailsOp: Operation<typeof input, GetJobDetailsResult> = {
  name: "get_job_details",
  domain: "jobs",
  result: { kind: "object", fields: ["job", "dataset_id", "job_id"] },
  summary:
    "Get job details for the job that produced a dataset. The job is read in full, so a failed " +
    "job's logs come with it: tool_stdout, tool_stderr, job_stdout, job_stderr, stdout and " +
    "stderr. A log longer than 4 KB keeps its first and last 2 KB, cut on line boundaries, with " +
    "a line saying how much was left out.",
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
