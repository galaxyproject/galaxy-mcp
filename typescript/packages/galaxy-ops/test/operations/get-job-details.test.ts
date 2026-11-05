import { describe, it, expect } from "vitest";
import { z } from "zod";
import { getJobDetailsOp, getJobDetails } from "../../src/operations/get-job-details";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { GalaxyConnectionError, GalaxyNotFoundError, GalaxyAuthError, GalaxyValidationError } from "../../src/errors";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_job_details", () => {
  it("(a) historyId path: reads creating_job from the dataset in that history; provenance not called", async () => {
    let provenanceCalls = 0;
    const client = mockClient({
      GET: (path, _init) => {
        if (path.includes("/provenance")) {
          provenanceCalls++;
          return { data: { job_id: "should-not-use" }, response: { status: 200 } };
        }
        if (path === "/api/datasets/{dataset_id}") {
          return { data: { id: "d1", history_id: "h1", creating_job: "j1" }, response: { status: 200 } };
        }
        if (path.includes("/jobs/")) {
          return { data: { id: "j1", state: "ok" }, response: { status: 200 } };
        }
        return { error: "not found", response: { status: 404 } };
      },
    });
    const out = await getJobDetails({ datasetId: "d1", historyId: "h1" }, ctxWith(client));
    expect(out.job_id).toBe("j1");
    expect(out.dataset_id).toBe("d1");
    expect((out.job as any).id).toBe("j1");
    expect((out.job as any).state).toBe("ok");
    expect(provenanceCalls).toBe(0);
  });

  it("(b) no historyId: reads creating_job from dataset GET; provenance not called", async () => {
    let provenanceCalls = 0;
    const client = mockClient({
      GET: (path, _init) => {
        if (path.includes("/provenance")) {
          provenanceCalls++;
          return { data: { job_id: "should-not-use" }, response: { status: 200 } };
        }
        if (path.includes("/datasets/") && !path.includes("/jobs/")) {
          return { data: { creating_job: "j2" }, response: { status: 200 } };
        }
        if (path.includes("/jobs/")) {
          return { data: { id: "j2" }, response: { status: 200 } };
        }
        return { error: "not found", response: { status: 404 } };
      },
    });
    const out = await getJobDetails({ datasetId: "d2" }, ctxWith(client));
    expect(out.job_id).toBe("j2");
    expect(out.dataset_id).toBe("d2");
    expect(provenanceCalls).toBe(0);
  });

  it("(c) error: dataset GET returns 404 -> throws", async () => {
    const client = mockClient({
      GET: (_path, _init) => {
        return { error: "not found", response: { status: 404 } };
      },
    });
    await expect(getJobDetails({ datasetId: "d3" }, ctxWith(client))).rejects.toThrow();
  });
});

// server.py, get_job_details: Galaxy finds the dataset by its id alone, so with a history named
// the dataset's own record decides whether it is in that history. The job's history does not:
// a dataset copied into this history keeps the job that made the original. The same record
// names the job, so the provenance route is never asked.
describe("get_job_details with a history named", () => {
  const D = "0123456789abcdef";
  const H = "1a2b3c4d5e6f7a8b";
  const OTHER = "8b7a6f5e4d3c2b1a";
  const J = "3333333333333333";
  const DATASET = "/api/datasets/{dataset_id}";
  const JOB = "/api/jobs/{job_id}";
  const REFUSAL =
    `Dataset ID '${D}' not found in history '${H}': Galaxy's record of that dataset does ` +
    "not put it in that history, so its job was not returned.";
  const inHistory = (historyId: string) => ({ id: D, history_id: historyId, creating_job: J });
  // Every reply that does not put the dataset in the history named, as server.py's tests
  // list them: another history, null (a dataset in no history), no history_id at all, and a
  // listing -- what a "." for an id folds the read into -- even one naming the right history.
  const NOT_THERE: Array<[string, unknown]> = [
    ["a record in another history", inHistory(OTHER)],
    ["a record in no history", { id: D, history_id: null, creating_job: J }],
    ["a record without history_id", { id: D, creating_job: J }],
    ["a listing", [inHistory(H)]],
  ];

  function galaxy(reply: unknown) {
    const paths: string[] = [];
    const client = mockClient({
      GET: (path) => {
        paths.push(path);
        if (path === DATASET) return { data: reply, response: { status: 200 } };
        return { data: { id: J, history_id: OTHER, state: "ok" }, response: { status: 200 } };
      },
    });
    return { ctx: ctxWith(client), paths };
  }

  it.each(NOT_THERE)("refuses %s before any job is read", async (_label, reply) => {
    const { ctx, paths } = galaxy(reply);
    const failure = await getJobDetails({ datasetId: D, historyId: H }, ctx).catch((e) => e);
    expect(failure).toBeInstanceOf(GalaxyNotFoundError);
    expect(failure.message).toBe(REFUSAL);
    expect(failure.http).toBeUndefined();
    expect(paths).toEqual([DATASET]);
  });

  it("reports a dataset record it could not read", async () => {
    const paths: string[] = [];
    const client = mockClient({
      GET: (path) => {
        paths.push(path);
        if (path === DATASET) return { error: { err_msg: "boom" }, response: { status: 500 } };
        return { data: { id: J }, response: { status: 200 } };
      },
    });
    const failure = await getJobDetails({ datasetId: D, historyId: H }, ctxWith(client)).catch(
      (e) => e,
    );
    expect(failure).toBeInstanceOf(GalaxyConnectionError);
    expect(failure.http?.status).toBe(500);
    expect(paths).toEqual([DATASET]);
  });

  it("answers for a copy in the history named, with the job that made the original", async () => {
    const { ctx, paths } = galaxy(inHistory(H));
    const out = await getJobDetails({ datasetId: D, historyId: H }, ctx);
    expect(out).toEqual({ job: { id: J, history_id: OTHER, state: "ok" }, dataset_id: D, job_id: J });
    expect(paths).toEqual([DATASET, JOB]);
  });

  it("does not hold the history's case against the caller", async () => {
    // Galaxy decodes either case and writes lowercase, so the two spell one history.
    const { ctx } = galaxy(inHistory(H));
    const out = await getJobDetails({ datasetId: D, historyId: H.toUpperCase() }, ctx);
    expect(out.job_id).toBe(J);
  });

  it("scopes nothing when no history is named: a dataset anywhere answers", async () => {
    const { ctx, paths } = galaxy(inHistory(OTHER));
    const out = await getJobDetails({ datasetId: D }, ctx);
    expect(out.job_id).toBe(J);
    expect(paths).toEqual([DATASET, JOB]);
  });
});

const J1 = "0123456789abcdef";

describe("get_job_details by job id", () => {
  it("is registered under the Python tool's name", () => {
    expect(getJobDetailsOp.name).toBe("get_job_details");
  });

  it("reads the job directly, plain and without a query, and names no dataset", async () => {
    const paths: string[] = [];
    const inits: any[] = [];
    const client = mockClient({
      GET: (path, init) => {
        paths.push(path);
        inits.push(init);
        return { data: { id: J1, state: "ok" }, response: { status: 200 } };
      },
    });
    const out = await getJobDetails({ jobId: J1 }, ctxWith(client));
    expect(paths).toEqual(["/api/jobs/{job_id}"]);
    expect(inits[0].params).toEqual({ path: { job_id: J1 } });
    expect(out).toEqual({ job: { id: J1, state: "ok" }, dataset_id: null, job_id: J1 });
  });

  it("sends no query on either path: the logs are get_job_logs' read, not this one's", async () => {
    const queries: unknown[] = [];
    const client = mockClient({
      GET: (path, init) => {
        if (path.includes("/jobs/")) {
          queries.push(init.params.query);
          return { data: { id: J1 }, response: { status: 200 } };
        }
        return { data: { creating_job: J1 }, response: { status: 200 } };
      },
    });
    await getJobDetails({ jobId: J1 }, ctxWith(client));
    await getJobDetails({ datasetId: "d1" }, ctxWith(client));
    expect(queries).toEqual([undefined, undefined]);
    expect(getJobDetailsOp.summary).toContain("one get_job_logs call away");
    // Not "get_job_logs(jobId)": the CLI would respell that to get_job_logs(--job-id), a
    // flag get_job_logs does not have (jobId is its positional).
    expect(getJobDetailsOp.summary).not.toContain("get_job_logs(");
    // Galaxy adds job_metrics only to a full read, and only for an admin: the plain read
    // this op makes never carries them, so the summary must not promise them.
    expect(getJobDetailsOp.summary).not.toContain("metrics");
  });

  it("words the message for the thing that was asked about", () => {
    const project = getJobDetailsOp.project!;
    expect(project({ job: {}, dataset_id: null, job_id: J1 }, {} as any)).toEqual({
      message: `Retrieved job details for job '${J1}'`,
    });
    expect(project({ job: {}, dataset_id: "d1", job_id: J1 }, {} as any)).toEqual({
      message: "Retrieved job details for dataset 'd1'",
    });
  });

  it.each([400, 404])("a %i from the jobs API is not found", async (status) => {
    const client = mockClient({
      GET: () => ({ error: { err_msg: "no" }, response: { status } }),
    });
    await expect(getJobDetails({ jobId: "0000000000000404" }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyNotFoundError,
    );
  });

  it("a 403 from the jobs API is still an auth failure", async () => {
    const client = mockClient({
      GET: () => ({ error: { err_msg: "no" }, response: { status: 403 } }),
    });
    await expect(getJobDetails({ jobId: "0000000000000403" }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyAuthError,
    );
  });

  it("refuses both ids, and neither, before any request", async () => {
    let calls = 0;
    const client = mockClient({
      GET: () => {
        calls++;
        return { data: {}, response: { status: 200 } };
      },
    });
    await expect(
      getJobDetails({ datasetId: "d1", jobId: J1 }, ctxWith(client)),
    ).rejects.toBeInstanceOf(GalaxyValidationError);
    await expect(getJobDetails({}, ctxWith(client))).rejects.toBeInstanceOf(GalaxyValidationError);
    await expect(
      getJobDetails({ datasetId: null, jobId: null }, ctxWith(client)),
    ).rejects.toThrow("neither was given");
    expect(calls).toBe(0);
  });

  it.each([".", "../histories", "not-an-id", "j0000001"])(
    "refuses %j as a job id before any request",
    async (value) => {
      // "." and "../histories" survive encoding and the fetch API folds them into the path:
      // /api/jobs/ and /api/histories, both of which answer 200 with a list. Galaxy would
      // answer "not-an-id" with a 400, which reads as not found anyway; the refusal just
      // happens before the request.
      let calls = 0;
      const client = mockClient({
        GET: () => {
          calls++;
          return { data: [{ id: "0000000000000001" }], response: { status: 200 } };
        },
      });
      const failure = await getJobDetails({ jobId: value }, ctxWith(client)).catch((e) => e);
      expect(failure).toBeInstanceOf(GalaxyNotFoundError);
      expect(failure.message).toBe(
        `Job ID '${value}' not found: that is not a Galaxy id. Galaxy's ids are hex strings, ` +
          "as list_jobs() reports them. Nothing was sent to Galaxy.",
      );
      expect(failure.http).toBeUndefined();
      expect(calls).toBe(0);
    },
  );

  it("refuses both ids before looking at the shape of either", async () => {
    const client = mockClient({ GET: () => ({ data: {}, response: { status: 200 } }) });
    await expect(getJobDetails({ datasetId: "d1", jobId: "." }, ctxWith(client))).rejects.toThrow(
      "not both",
    );
  });

  it.each([
    [{ id: "fedcba9876543210", state: "ok" }],
    [[{ id: J1 }]],
    [{ state: "ok" }],
  ])("refuses a 200 that is not this job's record: %j", async (answer) => {
    // A 200 is not the answer; a 200 for this id is. Loom's verifyGalaxyRun makes the same
    // check, so that a request landing on some other resource cannot certify a job.
    const client = mockClient({ GET: () => ({ data: answer, response: { status: 200 } }) });
    const failure = await getJobDetails({ jobId: J1 }, ctxWith(client)).catch((e) => e);
    expect(failure).toBeInstanceOf(GalaxyConnectionError);
    expect(failure.message).toBe(
      `Galaxy answered the read of job '${J1}' with something that is not that job's record ` +
        "(no matching id), so it was not returned.",
    );
  });

  it("does not hold the id's case against the caller", async () => {
    // Galaxy decodes either case and writes lowercase, so the two spell one record.
    const client = mockClient({ GET: () => ({ data: { id: J1 }, response: { status: 200 } }) });
    const out = await getJobDetails({ jobId: J1.toUpperCase() }, ctxWith(client));
    expect(out.job_id).toBe(J1.toUpperCase());
  });

  it("declares its inputs in the Python signature's order", () => {
    // dataset_id, history_id were the positional pair before job_id existed; job_id after
    // them is what keeps `get_job_details(dataset, history)` meaning what it did.
    expect(Object.keys(getJobDetailsOp.input)).toEqual(["datasetId", "historyId", "jobId"]);
  });

  it("mirrors the Python signature in its schema", () => {
    const schema = z.object(getJobDetailsOp.input);
    expect(schema.parse({})).toEqual({});
    expect(schema.parse({ datasetId: null, jobId: null, historyId: null })).toEqual({
      datasetId: null,
      jobId: null,
      historyId: null,
    });
  });
});
