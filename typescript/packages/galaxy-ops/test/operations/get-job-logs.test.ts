import { describe, it, expect } from "vitest";
import { z } from "zod";
import { getJobLogsOp, getJobLogs, JOB_LOG_FIELDS } from "../../src/operations/get-job-logs";
import { logEnds } from "../../src/operations/log-ends";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { GalaxyAuthError, GalaxyConnectionError, GalaxyNotFoundError, GalaxyValidationError } from "../../src/errors";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });
const J1 = "0123456789abcdef";

describe("get_job_logs", () => {
  it("is registered under the Python tool's name, as a read", () => {
    expect(getJobLogsOp.name).toBe("get_job_logs");
    expect(getJobLogsOp.readOnly).toBeUndefined();
    expect(getJobLogsOp.requires).toBeUndefined();
    expect(getJobLogsOp.summary).toContain("get_job_details");
  });

  it("reads the job in full and answers only the log fields, in Galaxy's order", async () => {
    const inits: any[] = [];
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/jobs/{job_id}");
        inits.push(init);
        // A finished job, as 26.2 answers one: all six fields as strings, the legacy pair
        // joined from the columns the way Job.stdout and Job.stderr do it. Sent out of
        // order here, so the order of the answer is seen to be the op's and not the reply's.
        return {
          data: {
            id: J1,
            state: "error",
            exit_code: 1,
            stderr: "",
            stdout: "a\nb\n\nran",
            tool_stdout: "a\nb\n",
            tool_stderr: "",
            job_stdout: "ran",
            job_stderr: "",
            job_metrics: [],
          },
          response: { status: 200 },
        };
      },
    });
    const out = await getJobLogs({ jobId: J1 }, ctxWith(client));
    expect(inits).toEqual([{ params: { path: { job_id: J1 }, query: { full: true } } }]);
    expect(out).toEqual({
      tool_stdout: "a\nb\n",
      tool_stderr: "",
      job_stdout: "ran",
      job_stderr: "",
      stdout: "a\nb\n\nran",
      stderr: "",
    });
    expect(Object.keys(out)).toEqual(["tool_stdout", "tool_stderr", "job_stdout", "job_stderr", "stdout", "stderr"]);
    expect(JOB_LOG_FIELDS).toEqual(["tool_stdout", "tool_stderr", "job_stdout", "job_stderr", "stdout", "stderr"]);
  });

  it("leaves out a null, a missing and a non-string field, and keeps an empty string", async () => {
    const client = mockClient({
      GET: () => ({
        data: { id: J1, state: "new", job_stdout: null, job_stderr: 7, stdout: ["x"], stderr: "" },
        response: { status: 200 },
      }),
    });
    expect(await getJobLogs({ jobId: J1 }, ctxWith(client))).toEqual({ stderr: "" });
  });

  it("cuts each field to the budget, and 0 leaves them whole", async () => {
    const log = Array.from({ length: 500 }, (_, i) => `line ${i}`).join("\n");
    const client = mockClient({
      GET: () => ({ data: { id: J1, tool_stderr: log, stderr: log }, response: { status: 200 } }),
    });
    const cut = await getJobLogs({ jobId: J1, logBytes: 100 }, ctxWith(client));
    expect(cut.tool_stderr).toBe(logEnds(log, 100));
    expect(cut.tool_stderr).toContain("bytes omitted ...]");
    const whole = await getJobLogs({ jobId: J1, logBytes: 0 }, ctxWith(client));
    expect(whole).toEqual({ tool_stderr: log, stderr: log });
    const byDefault = await getJobLogs({ jobId: J1 }, ctxWith(client));
    expect(byDefault.tool_stderr).toBe(logEnds(log, 4096));
  });

  it("answers a job that has not run with the empty streams Galaxy always sends", async () => {
    // view_show_job copies the four tool_*/job_* columns as they are (null before the job
    // has run) while Job.stdout and Job.stderr are properties that always answer a string,
    // so the emptiest answer a Galaxy gives is the two legacy fields as "", never {}.
    const client = mockClient({
      GET: () => ({
        data: { id: J1, state: "new", tool_stdout: null, tool_stderr: null, stdout: "", stderr: "" },
        response: { status: 200 },
      }),
    });
    expect(await getJobLogs({ jobId: J1 }, ctxWith(client))).toEqual({ stdout: "", stderr: "" });
    expect(getJobLogsOp.summary).toContain("always answers as a string");
    expect(getJobLogsOp.summary).not.toContain("has not written yet");
  });

  it("words the message for the job, whatever was there", () => {
    const project = getJobLogsOp.project!;
    expect(project({ tool_stdout: "x" }, { jobId: J1, logBytes: 4096 })).toEqual({
      message: `Retrieved job logs for job '${J1}'`,
    });
    expect(project({}, { jobId: J1, logBytes: 4096 })).toEqual({
      message: `Retrieved job logs for job '${J1}'`,
    });
  });

  it.each([-1, 1.5])("refuses a budget of %s before any request", async (logBytes) => {
    let calls = 0;
    const client = mockClient({
      GET: () => {
        calls++;
        return { data: { id: J1 }, response: { status: 200 } };
      },
    });
    const failure = await getJobLogs({ jobId: J1, logBytes }, ctxWith(client)).catch((e) => e);
    expect(failure).toBeInstanceOf(GalaxyValidationError);
    expect(failure.message).toBe(`log_bytes must be 0 or greater (got ${logBytes})`);
    expect(calls).toBe(0);
  });

  it.each([".", "../histories", "not-an-id"])("refuses %j as a job id before any request", async (value) => {
    let calls = 0;
    const client = mockClient({
      GET: () => {
        calls++;
        return { data: [{ id: J1 }], response: { status: 200 } };
      },
    });
    const failure = await getJobLogs({ jobId: value }, ctxWith(client)).catch((e) => e);
    expect(failure).toBeInstanceOf(GalaxyNotFoundError);
    expect(failure.message).toBe(
      `Job ID '${value}' not found: that is not a Galaxy id. Galaxy's ids are hex strings, ` +
        "as list_jobs() reports them. Nothing was sent to Galaxy.",
    );
    expect(calls).toBe(0);
  });

  it("checks the budget before the id, as the Python tool does", async () => {
    const client = mockClient({ GET: () => ({ data: {}, response: { status: 200 } }) });
    await expect(getJobLogs({ jobId: ".", logBytes: -1 }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyValidationError,
    );
  });

  it.each([400, 404])("a %i from the jobs API is not found", async (status) => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "no" }, response: { status } }) });
    await expect(getJobLogs({ jobId: "0000000000000404" }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyNotFoundError,
    );
  });

  it("a 403 from the jobs API is still an auth failure", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "no" }, response: { status: 403 } }) });
    await expect(getJobLogs({ jobId: "0000000000000403" }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyAuthError,
    );
  });

  it.each([[{ id: "fedcba9876543210", tool_stderr: "x".repeat(10000) }], [[{ id: J1 }]], [{ state: "ok" }]])(
    "refuses a 200 that is not this job's record: %j",
    async (answer) => {
      const client = mockClient({ GET: () => ({ data: answer, response: { status: 200 } }) });
      const failure = await getJobLogs({ jobId: J1 }, ctxWith(client)).catch((e) => e);
      expect(failure).toBeInstanceOf(GalaxyConnectionError);
      expect(failure.message).toBe(
        `Galaxy answered the read of job '${J1}' with something that is not that job's record ` +
          "(no matching id), so it was not returned.",
      );
    },
  );

  it("does not hold the id's case against the caller", async () => {
    const client = mockClient({ GET: () => ({ data: { id: J1, stdout: "s" }, response: { status: 200 } }) });
    expect(await getJobLogs({ jobId: J1.toUpperCase() }, ctxWith(client))).toEqual({ stdout: "s" });
  });

  it("words its failures from the Python tool's own sentences", () => {
    const failure = getJobLogsOp.failure!;
    expect(failure.shape).toBe("raise-for-status");
    expect(failure.action).toBe("Get job logs");
    expect(failure.context!({ jobId: J1, logBytes: 4096 })).toEqual({ job_id: J1 });
    for (const status of [400, 404]) {
      expect(failure.sentence!("", status, { jobId: J1, logBytes: 4096 })).toBe(
        `Job ID '${J1}' not found or not accessible. Make sure the job exists and you have permission to view it.`,
      );
    }
    expect(failure.sentence!("", 500, { jobId: J1, logBytes: 4096 })).toBeUndefined();
  });

  it("mirrors the Python signature in its schema", () => {
    expect(Object.keys(getJobLogsOp.input)).toEqual(["jobId", "logBytes"]);
    const schema = z.object(getJobLogsOp.input);
    expect(schema.parse({ jobId: J1 })).toEqual({ jobId: J1, logBytes: 4096 });
    expect(schema.parse({ jobId: J1, logBytes: 0 })).toEqual({ jobId: J1, logBytes: 0 });
    expect(schema.safeParse({}).success).toBe(false);
    expect(schema.safeParse({ jobId: J1, logBytes: null }).success).toBe(false);
    expect(schema.safeParse({ jobId: J1, logBytes: 1.5 }).success).toBe(false);
    expect(schema.safeParse({ jobId: J1, logBytes: "5" }).success).toBe(false);
  });
});
