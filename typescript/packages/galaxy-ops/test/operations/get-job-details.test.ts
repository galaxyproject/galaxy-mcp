import { describe, it, expect } from "vitest";
import { z } from "zod";
import { getJobDetailsOp, getJobDetails } from "../../src/operations/get-job-details";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { GalaxyConnectionError, GalaxyNotFoundError, GalaxyAuthError, GalaxyValidationError } from "../../src/errors";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_job_details", () => {
  it("(a) historyId path: resolves job_id from provenance then fetches job", async () => {
    const client = mockClient({
      GET: (path, _init) => {
        if (path.includes("/provenance")) {
          return { data: { job_id: "j1" }, response: { status: 200 } };
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
  });

  it("(b) fallback path: no historyId, reads creating_job from dataset GET; provenance not called", async () => {
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

  // server.py, get_job_details: a provenance failure is held, not raised. The dataset's own
  // record names the job that made it, so a history whose provenance is out of reach still
  // answers the question -- and the held failure is only reported if the fallback fails too,
  // or finds no job. That is what the other server does, and it is why this asks for the job
  // rather than refusing.
  it("(d) historyId given, provenance returns 403 -> the dataset's creating_job answers", async () => {
    const client = mockClient({
      GET: (path, _init) => {
        if (path.includes("/provenance")) {
          return { error: { err_msg: "Forbidden" }, response: { status: 403 } };
        }
        if (path.includes("/datasets/")) {
          return { data: { creating_job: "j4" }, response: { status: 200 } };
        }
        return { data: { id: "j4", state: "ok" }, response: { status: 200 } };
      },
    });
    const out = await getJobDetails({ datasetId: "d4", historyId: "h4" }, ctxWith(client));
    expect(out.job_id).toBe("j4");
  });

  it("(d2) a held provenance failure is what a dataset with no creating_job reports", async () => {
    const client = mockClient({
      GET: (path, _init) => {
        if (path.includes("/provenance")) {
          return { error: { err_msg: "Forbidden" }, response: { status: 403 } };
        }
        return { data: { id: "d5" }, response: { status: 200 } };
      },
    });
    await expect(
      getJobDetails({ datasetId: "d5", historyId: "h5" }, ctxWith(client)),
    ).rejects.toBeInstanceOf(GalaxyAuthError);
  });

  it("(e) historyId given, provenance returns 200 but no job_id -> falls through to dataset creating_job", async () => {
    const client = mockClient({
      GET: (path, _init) => {
        if (path.includes("/provenance")) {
          // 200 response with no job_id field
          return { data: { some_other_field: "x" }, response: { status: 200 } };
        }
        if (path.includes("/datasets/") && !path.includes("/jobs/")) {
          return { data: { creating_job: "j5" }, response: { status: 200 } };
        }
        if (path.includes("/jobs/")) {
          return { data: { id: "j5", state: "ok" }, response: { status: 200 } };
        }
        return { error: "not found", response: { status: 404 } };
      },
    });
    const out = await getJobDetails({ datasetId: "d5", historyId: "h5" }, ctxWith(client));
    expect(out.job_id).toBe("j5");
    expect(out.dataset_id).toBe("d5");
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
    expect(getJobDetailsOp.summary).toContain("get_job_logs");
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
