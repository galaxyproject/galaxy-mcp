import { z } from "zod";
import type { GalaxyContext } from "../context";
import { httpError, GalaxyConnectionError, GalaxyNotFoundError, GalaxyValidationError } from "../errors";
import { isEncodedId, isThisRecord, notAGalaxyId, notThatRecord } from "./encoded-id";
import { jobLookupFailure, jobNotFoundSentence } from "./jobs-common";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

// The dataset endpoint's creating_job field is not always typed; hand-type for safety.
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
  /** The dataset the job was found through, or null when the job was asked for by id. */
  dataset_id: string | null;
  job_id: string;
}

// In the Python signature's order: dataset_id and history_id were the two positional
// parameters before job_id existed, and keeping job_id after them is what keeps a caller's
// positional history id from landing in it.
const input = {
  datasetId: z
    .string()
    .nullish()
    .describe("dataset (HDA) id whose creating job to look up; give this or jobId, not both"),
  historyId: z
    .string()
    .nullish()
    .describe(
      "history id containing the dataset; a dataset in any other history is refused. Only " +
        "used with datasetId",
    ),
  jobId: z
    .string()
    .nullish()
    .describe("job id to look up directly; give this or datasetId, not both"),
};
type In = {
  datasetId?: string | null;
  historyId?: string | null;
  jobId?: string | null;
};

/**
 * server.py, _is_in_history: whether a dataset's own record puts it in the history asked
 * about. Galaxy writes history_id into every view of a history dataset, so a record naming
 * another history, or none, does not say the dataset is in this one; compared the way
 * isThisRecord compares an id, since Galaxy writes lowercase hex and decodes either case.
 */
function isInHistory(record: unknown, historyId: string): boolean {
  if (!record || typeof record !== "object" || Array.isArray(record)) return false;
  const held = (record as { history_id?: unknown }).history_id;
  return typeof held === "string" && held.toLowerCase() === historyId.toLowerCase();
}

/** server.py, _not_in_history: the Python server's sentence, byte for byte, so change both. */
function notInHistory(datasetId: string, historyId: string): string {
  return (
    `Dataset ID '${datasetId}' not found in history '${historyId}': Galaxy's record of ` +
    "that dataset does not put it in that history, so its job was not returned."
  );
}

async function run(i: In, ctx: GalaxyContext): Promise<GetJobDetailsResult> {
  // server.py: refused before anything is asked of Galaxy, and worded there, so neither
  // sentence carries a status. Truthiness, as over there: an empty id is no id.
  if (i.datasetId && i.jobId) {
    throw new GalaxyValidationError(
      "get_job_details takes a dataset_id or a job_id, not both " +
        `(got dataset_id '${i.datasetId}' and job_id '${i.jobId}').`,
    );
  }
  if (!i.datasetId && !i.jobId) {
    throw new GalaxyValidationError(
      "get_job_details needs a dataset_id or a job_id; neither was given.",
    );
  }

  if (i.jobId) {
    // Not an id, not a request: see encoded-id.ts for what a stray "." would turn into.
    // A refusal of our own, worded whole, so no facts for the failure contract.
    if (!isEncodedId(i.jobId)) {
      throw new GalaxyNotFoundError(notAGalaxyId("Job", i.jobId, "list_jobs()"));
    }
    const { data, error, response } = await ctx.client.GET("/api/jobs/{job_id}", {
      params: { path: { job_id: i.jobId } },
    });
    if (error || !data) throw jobLookupFailure(response, error);
    // A 200 is not the answer; a 200 for this job is. The caller holding the id is asking
    // whether that job exists, and a reply about anything else -- a listing, a redirect's
    // target -- must not be handed back as it.
    if (!isThisRecord(data, i.jobId)) {
      throw new GalaxyConnectionError(notThatRecord("job", i.jobId), response.status);
    }
    return { job: data as JobDetail, dataset_id: null, job_id: i.jobId };
  }

  const datasetId = i.datasetId as string;
  // server.py, get_job_details: the dataset's own record answers both questions, which
  // history it is in and which job made it. Galaxy finds the dataset by its id alone, so a
  // dataset in any other history answers too, and the record's history_id is the one thing
  // that says where it is. Its creating_job is the job Galaxy's provenance route reports as
  // well, so that route is not asked: it would add a request and nothing else. The job is no
  // test of the history: a dataset copied into this history keeps the job that made the
  // original, which ran in another one.
  const { data, error, response } = await ctx.client.GET("/api/datasets/{dataset_id}", {
    params: { path: { dataset_id: datasetId } },
  });
  if (error || !data) throw httpError(response, error);
  // A refusal of our own, worded whole, so no facts for the failure contract.
  if (i.historyId && !isInHistory(data, i.historyId)) {
    throw new GalaxyNotFoundError(notInHistory(datasetId, i.historyId));
  }
  const jobId = (data as DatasetMeta).creating_job;
  if (!jobId) {
    throw new GalaxyNotFoundError(
      `No job information found for dataset '${datasetId}'. ` +
        "The dataset may not have been created by a job.",
    );
  }

  const { data: jobData, error: jobError, response: jobResp } = await ctx.client.GET("/api/jobs/{job_id}", {
    params: { path: { job_id: jobId } },
  });
  if (jobError || !jobData) throw httpError(jobResp, jobError);

  return {
    job: jobData as JobDetail,
    dataset_id: datasetId,
    job_id: jobId,
  };
}

export const getJobDetailsOp: Operation<typeof input, GetJobDetailsResult> = {
  name: "get_job_details",
  domain: "jobs",
  // server.py's docstring, so the two surfaces describe the read in one voice: what the
  // record carries, and that the logs are get_job_logs' question rather than this one's.
  // That pointer names no parameter on purpose: spellParamNames respells every key of THIS
  // op's input in the surface's own spelling, and the CLI has no --job-id on get_job_logs.
  summary:
    "Get a job's record, by its own id or by a dataset it created. The record carries the " +
    "job's state, exit code, tool id and version, create and update times, params, inputs " +
    "and outputs. It does not carry the job's stdout or stderr: a failed job's logs are one " +
    "get_job_logs call away. Exactly one of datasetId and jobId is required.",
  input,
  run,
  // server.py, get_job_details: the thing that was asked about -- the dataset, or the job
  // when it was asked for by id -- not what was found for it; the job's id is in `data`.
  project: (out) => ({
    message:
      out.dataset_id === null
        ? `Retrieved job details for job '${out.job_id}'`
        : `Retrieved job details for dataset '${out.dataset_id}'`,
  }),
  // server.py, _job_details_failed and _job_lookup_failed: every failure here is described
  // from the exception that actually failed. Through a dataset, a 404 keeps the tool's own
  // sentence because a 404 from the jobs API is as likely to be a permission problem as a
  // missing dataset. By job id, a 400 and a 404 share one sentence about the job -- see
  // jobLookupFailure -- and the context names the id the caller gave. The two routes are
  // asked through two different clients over there -- bioblend for the dataset and
  // provenance, requests itself for the job -- so they do not word a failure the same way.
  failure: {
    shape: (facts) => (facts.url.includes("/api/jobs/") ? "raise-for-status" : "bioblend-get"),
    action: "Get job details",
    context: (i) => (i.jobId ? { job_id: i.jobId } : { dataset_id: i.datasetId }),
    sentence: (_text, status, i) => {
      if (i.jobId) {
        return status === 400 || status === 404 ? jobNotFoundSentence(i.jobId) : undefined;
      }
      return status === 404
        ? `Dataset ID '${i.datasetId}' not found or job not accessible. ` +
            "Make sure the dataset exists and you have permission to view it."
        : undefined;
    },
  },
};

register(getJobDetailsOp as AnyOperation);

export const getJobDetails = (i: In, ctx: GalaxyContext) => runOperation(getJobDetailsOp, i, ctx);
