import { describe, it, expect } from "vitest";
import { executeToolRequest } from "../src/execute-tool-request";
import { mockClient } from "./util/mock-client";
import { DEFAULT_POLL } from "../src/context";
import { ToolRequestRejectedError, JobFailedError } from "../src/errors";
import type { GalaxyContext } from "../src/context";

const fastPoll = { ...DEFAULT_POLL, intervalMs: 0, maxIntervalMs: 0, jitter: 0, timeoutMs: 5000 };

/** Build a client scripted by path. `stateSeq` drives /state; `job` drives /jobs/{id}. */
function scripted(opts: {
  postJobs?: (body: any) => any;
  stateSeq: string[];
  detail?: any;
  jobStates?: string[];
}): GalaxyContext {
  let si = 0;
  let ji = 0;
  const client = mockClient({
    POST: (path, init) => {
      expect(path).toBe("/api/jobs");
      const body = init.body;
      // forbidden-pattern guard: no flat pipe-keys anywhere in inputs
      expect(JSON.stringify(body.inputs)).not.toMatch(/\|/);
      expect(body.strict).toBe(true);
      return opts.postJobs
        ? { data: opts.postJobs(body), response: { status: 200 } }
        : { data: { tool_request_id: "tr1", task_result: { id: "t" } }, response: { status: 200 } };
    },
    GET: (path) => {
      if (path.endsWith("/state")) {
        const s = opts.stateSeq[Math.min(si, opts.stateSeq.length - 1)]!;
        si++;
        return { data: s, response: { status: 200 } };
      }
      if (path === "/api/tool_requests/{id}") {
        return { data: opts.detail, response: { status: 200 } };
      }
      // /api/jobs/{job_id}
      const st = opts.jobStates?.[Math.min(ji, (opts.jobStates?.length ?? 1) - 1)] ?? "ok";
      ji++;
      return { data: { id: "j1", state: st }, response: { status: 200 } };
    },
  });
  return { client, poll: fastPoll };
}

const goodInputs = { input_file: { src: "hda", id: "d1" } };

describe("executeToolRequest", () => {
  it("new -> submitted -> jobs ok yields state 'ok'", async () => {
    const ctx = scripted({
      stateSeq: ["new", "submitted"],
      detail: { id: "tr1", state: "submitted", jobs: [{ src: "job", id: "j1" }], implicit_collections: [] },
      jobStates: ["running", "ok"],
    });
    const run = await executeToolRequest({ toolId: "fastqc/0.74", historyId: "h1", inputs: goodInputs }, ctx);
    expect(run.state).toBe("ok");
    expect(run.toolRequestId).toBe("tr1");
    expect(run.jobs).toHaveLength(1);
  });

  it("state 'failed' throws ToolRequestRejectedError with err_msg", async () => {
    const ctx = scripted({
      stateSeq: ["new", "failed"],
      detail: { id: "tr1", state: "failed", state_message: { err_msg: "bad param" }, jobs: [], implicit_collections: [] },
    });
    await expect(
      executeToolRequest({ toolId: "fastqc/0.74", historyId: "h1", inputs: goodInputs }, ctx),
    ).rejects.toMatchObject({ constructor: ToolRequestRejectedError, errMsg: "bad param" });
  });

  it("a failed job throws JobFailedError", async () => {
    const ctx = scripted({
      stateSeq: ["submitted"],
      detail: { id: "tr1", state: "submitted", jobs: [{ src: "job", id: "j1" }], implicit_collections: [] },
      jobStates: ["error"],
    });
    await expect(
      executeToolRequest({ toolId: "fastqc/0.74", historyId: "h1", inputs: goodInputs }, ctx),
    ).rejects.toBeInstanceOf(JobFailedError);
  });
});
