import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyNotFoundError, GalaxyValidationError } from "../errors";
import { legacyGet } from "../legacy";
import { isEncodedId, isThisRecord, notAGalaxyId, notThatRecord } from "./encoded-id";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

const MIB = 1024 * 1024;
const TERMINAL_STATES = new Set(["ok", "error", "failed", "paused", "deleted", "stopped", "skipped"]);
const PEAK_MEMORY_NAMES = ["memory.peak", "memory.max_usage_in_bytes"];
const OOM_NAMES = [
  "memory.events.oom_kill",
  "memory.events.oom_group_kill",
  "memory.events.oom",
  "memory.failcnt",
];

interface ToolLookup {
  tool_id?: string;
  representation?: Record<string, unknown>;
}

interface JobRecord {
  id?: string;
  tool_id?: string;
  state?: string;
}

interface JobMetric {
  plugin?: string;
  name?: string;
  raw_value?: string;
}

export interface ResourcePreflightReport {
  job_id: string | null;
  state: string | null;
  observed: Record<string, number | null>;
  declared: Record<string, number | null>;
  recommendation: Record<string, number | null>;
  safety_factor: number;
  warnings: Array<{ code: string; message: string }>;
}

const input = {
  toolUuid: z.string().describe("UUID of the user-defined tool"),
  jobId: z.string().describe("completed representative job id returned by run_user_tool"),
  safetyFactor: z.coerce.number().default(1.5).describe("margin applied to observed memory and runtime; at least 1"),
};
type In = {
  toolUuid: string;
  jobId: string;
  safetyFactor?: number;
};

const number = (value: unknown): number | null => {
  const result = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  return Number.isFinite(result) ? result : null;
};

function resourceRequirements(representation: Record<string, unknown>): Record<string, unknown> {
  const requirements = Array.isArray(representation.requirements) ? representation.requirements : [];
  return (requirements.find(
    (requirement) =>
      typeof requirement === "object" && requirement !== null && (requirement as Record<string, unknown>).type === "resource",
  ) ?? {}) as Record<string, unknown>;
}

function evaluate(
  representation: Record<string, unknown>,
  job: JobRecord,
  metrics: Array<JobMetric | null>,
  i: In,
): ResourcePreflightReport {
  const safetyFactor = i.safetyFactor ?? 1.5;
  if (!Number.isFinite(safetyFactor) || safetyFactor < 1) {
    throw new GalaxyValidationError("safety_factor must be a finite number at least 1");
  }
  if (!job.state || !TERMINAL_STATES.has(job.state)) {
    throw new GalaxyValidationError(`Job '${job.id}' is not finished (state: ${job.state})`);
  }
  const values = new Map<string, number>();
  for (const metric of metrics) {
    if (metric === null) continue;
    const value = number(metric.raw_value);
    if (metric.plugin && metric.name && value !== null) values.set(`${metric.plugin}:${metric.name}`, value);
  }
  const peakMemory = PEAK_MEMORY_NAMES.map((name) => values.get(`cgroup:${name}`)).find((value) => value !== undefined) ?? null;
  const runtime = values.get("core:runtime_seconds") ?? null;
  const oomEvents = Math.max(0, ...OOM_NAMES.map((name) => values.get(`cgroup:${name}`) ?? 0));
  const requirements = resourceRequirements(representation);
  const declared = Object.fromEntries(
    ["ram_min", "ram_max", "timelimit"].map((key) => [
      key,
      number(requirements[key]),
    ]),
  ) as Record<string, number | null>;
  const recommendation = {
    ram_min: peakMemory === null ? null : Math.ceil((peakMemory / MIB) * safetyFactor),
    timelimit: runtime === null ? null : Math.ceil(runtime * safetyFactor),
  };
  const ramMin = declared.ram_min ?? null;
  const ramMax = declared.ram_max ?? null;
  const timelimit = declared.timelimit ?? null;
  const warnings: Array<{ code: string; message: string }> = [];
  if (peakMemory === null) warnings.push({ code: "peak_memory_unavailable", message: "The server did not expose a cgroup peak-memory metric." });
  if (runtime === null) warnings.push({ code: "runtime_unavailable", message: "The server did not expose the job runtime metric." });
  if (job.state !== "ok") warnings.push({ code: "test_run_failed", message: `The test job finished in state '${job.state}'.` });
  if (oomEvents > 0) warnings.push({ code: "oom_observed", message: `The test run recorded ${oomEvents} OOM or allocation-failure events.` });
  if (Object.keys(requirements).length === 0) warnings.push({ code: "resource_requirement_missing", message: "The UDT has no resource requirement block." });
  if (recommendation.ram_min !== null && (ramMin === null || ramMin < recommendation.ram_min)) {
    warnings.push({ code: "ram_min_underestimated", message: `Declare ram_min of at least ${recommendation.ram_min} MiB for the tested scale.` });
  }
  if (recommendation.ram_min !== null && ramMax !== null && ramMax < recommendation.ram_min) {
    warnings.push({ code: "ram_range_conflict", message: "The recommended ram_min exceeds the declared ram_max." });
  }
  if (recommendation.timelimit !== null && (timelimit === null || timelimit < recommendation.timelimit)) {
    warnings.push({ code: "timelimit_underestimated", message: `Declare timelimit of at least ${recommendation.timelimit} seconds for the tested scale.` });
  }
  return {
    job_id: job.id ?? null,
    state: job.state ?? null,
    observed: {
      peak_memory_bytes: peakMemory,
      runtime_seconds: runtime,
      oom_events: oomEvents,
    },
    declared,
    recommendation,
    safety_factor: safetyFactor,
    warnings,
  };
}

async function run(i: In, ctx: GalaxyContext): Promise<ResourcePreflightReport> {
  if (!isEncodedId(i.jobId)) throw new GalaxyNotFoundError(notAGalaxyId("Job", i.jobId, "list_jobs()"));
  const tool = await legacyGet<ToolLookup>(ctx, "/api/unprivileged_tools/{tool_uuid}", {
    params: { path: { tool_uuid: i.toolUuid } },
  });
  if (!tool.tool_id) throw new GalaxyNotFoundError(`No user-defined tool found with UUID '${i.toolUuid}'`);
  const job = await legacyGet<JobRecord>(ctx, "/api/jobs/{job_id}", { params: { path: { job_id: i.jobId } } });
  if (!isThisRecord(job, i.jobId)) throw new GalaxyValidationError(notThatRecord("job", i.jobId));
  if (job.tool_id !== tool.tool_id) {
    throw new GalaxyValidationError(`Job '${i.jobId}' did not run user-defined tool '${i.toolUuid}'`);
  }
  const metrics = await legacyGet<Array<JobMetric | null>>(ctx, "/api/jobs/{job_id}/metrics", {
    params: { path: { job_id: i.jobId } },
  });
  if (!Array.isArray(metrics)) throw new GalaxyValidationError(`Galaxy returned invalid metrics for job '${i.jobId}'`);
  return evaluate(tool.representation ?? {}, job, metrics, i);
}

export const evaluateUserToolPreflightOp: Operation<typeof input, ResourcePreflightReport> = {
  name: "evaluate_user_tool_preflight",
  domain: "userTools",
  summary:
    "Evaluate a completed UDT run against its resource block. Recommendations apply only to the tested workload.",
  input,
  run,
  project: (_out, i) => ({ message: `Evaluated resource preflight for user-defined tool '${i.toolUuid}' from job '${i.jobId}'` }),
  failure: {
    shape: "raise-for-status",
    action: "Evaluate user tool preflight",
    context: (i) => ({ tool_uuid: i.toolUuid, job_id: i.jobId }),
  },
};

register(evaluateUserToolPreflightOp as AnyOperation);

export const evaluateUserToolPreflight = (i: In, ctx: GalaxyContext) =>
  runOperation(evaluateUserToolPreflightOp, { ...i, safetyFactor: i.safetyFactor ?? 1.5 }, ctx);
