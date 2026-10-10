import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyConnectionError, GalaxyNotFoundError, GalaxyValidationError } from "../errors";
import { isEncodedId, isThisRecord, notAGalaxyId, notThatRecord } from "./encoded-id";
import { jobLookupFailure, jobNotFoundSentence } from "./jobs-common";
import { logEnds } from "./log-ends";
import { register, runOperation } from "./registry";
import type { AnyOperation, InputOf, Operation } from "./types";

/** The log fields Galaxy adds to a job read with ?full=true, in the order they are answered. */
export const JOB_LOG_FIELDS = ["tool_stdout", "tool_stderr", "job_stdout", "job_stderr", "stdout", "stderr"] as const;
const DEFAULT_LOG_BYTES = 4096;

/** The present log fields, each already cut to the budget; a field Galaxy did not write is absent. */
export type JobLogs = Partial<Record<(typeof JOB_LOG_FIELDS)[number], string>>;

const input = {
  jobId: z.string().describe("job id, as run_tool, list_jobs or get_job_details report it"),
  logBytes: z
    .number()
    .int()
    .default(DEFAULT_LOG_BYTES)
    .describe(
      "budget per log field in UTF-8 bytes; a longer log keeps its first and last half on line " +
        "boundaries with a line saying how much was omitted; 0 returns every log uncut; negative is refused",
    ),
};
type In = { jobId: string; logBytes?: number };

async function run(i: In, ctx: GalaxyContext): Promise<JobLogs> {
  const logBytes = i.logBytes ?? DEFAULT_LOG_BYTES;
  // Refused before anything is sent, and worded whole, as the other reads by id do. The
  // integer check is for a programmatic caller who came in past zod, as pagination.ts does.
  if (!Number.isInteger(logBytes) || logBytes < 0) {
    throw new GalaxyValidationError(`log_bytes must be 0 or greater (got ${logBytes})`);
  }
  if (!isEncodedId(i.jobId)) {
    throw new GalaxyNotFoundError(notAGalaxyId("Job", i.jobId, "list_jobs()"));
  }
  const { data, error, response } = await ctx.client.GET("/api/jobs/{job_id}", {
    params: { path: { job_id: i.jobId }, query: { full: true } },
  });
  if (error || !data) throw jobLookupFailure(response, error);
  // The same check get_job_details makes: a 200 for this job, not merely a 200, and the
  // cut never runs over a body that is not this job's.
  if (!isThisRecord(data, i.jobId)) {
    throw new GalaxyConnectionError(notThatRecord("job", i.jobId), response.status);
  }
  // Only a string is a log. A missing key, a null (what Galaxy serialises for the tool_*
  // and job_* fields before the job has run) or anything else is left out, never answered
  // as "" -- an empty string is kept, because that is Galaxy saying the stream was empty,
  // and stdout and stderr are always a string over there, empty before the job has run.
  const record = data as Record<string, unknown>;
  const logs: JobLogs = {};
  for (const field of JOB_LOG_FIELDS) {
    const value = record[field];
    if (typeof value === "string") logs[field] = logEnds(value, logBytes);
  }
  return logs;
}

export const getJobLogsOp: Operation<typeof input, JobLogs> = {
  name: "get_job_logs",
  domain: "jobs",
  summary:
    "Read a job's logs: the stdout and stderr Galaxy keeps for it, each cut to a byte budget. " +
    "One read of the job in full, answered with only the six log fields that adds -- " +
    "tool_stdout, tool_stderr, job_stdout, job_stderr, stdout, stderr -- and nothing the " +
    "plain get_job_details read already has. The four tool_* and job_* fields are null until " +
    "the job has written them, and are then left out rather than returned empty; stdout and " +
    "stderr are the older combined form of those pairs, which Galaxy always answers as a " +
    "string, so a job that has not run reads as an empty stdout and stderr, and on a current " +
    "Galaxy the same text can appear under two names. A log longer than logBytes keeps its " +
    "first and last half, cut on line boundaries, with one line in the middle saying how " +
    "many of how many bytes were omitted, so a cut field carries at most logBytes of the log " +
    "plus that line.",
  input,
  run,
  // server.py, get_job_logs: the job that was asked about. Nothing is said about how much
  // was there -- data carries exactly the present fields, and Galaxy always answers stdout
  // and stderr, so a record with no logs at all is not an answer it gives.
  project: (_out, i) => ({ message: `Retrieved job logs for job '${i.jobId}'` }),
  // server.py, _job_lookup_failed with action "Get job logs": a 400 and a 404 share one
  // sentence about the job (see jobLookupFailure), and any other status goes through
  // format_error under this tool's own action. The read is a raw requests call over there.
  failure: {
    shape: "raise-for-status",
    action: "Get job logs",
    context: (i) => ({ job_id: i.jobId }),
    sentence: (_text, status, i) =>
      status === 400 || status === 404 ? jobNotFoundSentence(i.jobId) : undefined,
  },
};

register(getJobLogsOp as AnyOperation);

// The return type is the check: add a .default() above without one here and this stops
// compiling.
const withDefaults = (i: In): InputOf<typeof input> => ({
  ...i,
  logBytes: i.logBytes ?? DEFAULT_LOG_BYTES,
});

export const getJobLogs = (i: In, ctx: GalaxyContext) => runOperation(getJobLogsOp, withDefaults(i), ctx);
