import { describe, it, expect } from "vitest";
import { z } from "zod";
import { getJobDetailsOp, getJobDetails } from "../../src/operations/get-job-details";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { GalaxyNotFoundError, GalaxyAuthError, GalaxyValidationError } from "../../src/errors";

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

describe("get_job_details by job id", () => {
  it("is registered under the Python tool's name", () => {
    expect(getJobDetailsOp.name).toBe("get_job_details");
  });

  it("reads the job directly, sends no query, and names no dataset", async () => {
    const paths: string[] = [];
    const inits: any[] = [];
    const client = mockClient({
      GET: (path, init) => {
        paths.push(path);
        inits.push(init);
        return { data: { id: "j1", state: "ok" }, response: { status: 200 } };
      },
    });
    const out = await getJobDetails({ jobId: "j1" }, ctxWith(client));
    expect(paths).toEqual(["/api/jobs/{job_id}"]);
    expect(inits[0].params).toEqual({ path: { job_id: "j1" }, query: undefined });
    expect(out).toEqual({ job: { id: "j1", state: "ok" }, dataset_id: null, job_id: "j1" });
  });

  it("sends full=true when asked, on either path", async () => {
    const queries: unknown[] = [];
    const client = mockClient({
      GET: (path, init) => {
        if (path.includes("/jobs/")) {
          queries.push(init.params.query);
          return { data: { id: "j1" }, response: { status: 200 } };
        }
        return { data: { creating_job: "j1" }, response: { status: 200 } };
      },
    });
    await getJobDetails({ jobId: "j1", full: true }, ctxWith(client));
    await getJobDetails({ datasetId: "d1", full: true }, ctxWith(client));
    expect(queries).toEqual([{ full: true }, { full: true }]);
  });

  it("describes full by what it adds, not by fields the plain read already has", () => {
    // EncodedJobDetails (the non-full reply) already requires params, inputs and outputs;
    // full adds stdout/stderr, job messages, dependencies and metrics.
    const description = getJobDetailsOp.input.full.description ?? "";
    for (const added of ["stdout", "stderr", "job messages", "dependencies", "job metrics"]) {
      expect(description).toContain(added);
    }
    expect(description).toContain("params, inputs and outputs are in the plain read already");
  });

  it("words the message for the thing that was asked about", () => {
    const project = getJobDetailsOp.project!;
    expect(project({ job: {}, dataset_id: null, job_id: "j1" }, {} as any)).toEqual({
      message: "Retrieved job details for job 'j1'",
    });
    expect(project({ job: {}, dataset_id: "d1", job_id: "j1" }, {} as any)).toEqual({
      message: "Retrieved job details for dataset 'd1'",
    });
  });

  it.each([400, 404])("a %i from the jobs API is not found", async (status) => {
    const client = mockClient({
      GET: () => ({ error: { err_msg: "no" }, response: { status } }),
    });
    await expect(getJobDetails({ jobId: "j404" }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyNotFoundError,
    );
  });

  it("a 403 from the jobs API is still an auth failure", async () => {
    const client = mockClient({
      GET: () => ({ error: { err_msg: "no" }, response: { status: 403 } }),
    });
    await expect(getJobDetails({ jobId: "j403" }, ctxWith(client))).rejects.toBeInstanceOf(
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
      getJobDetails({ datasetId: "d1", jobId: "j1" }, ctxWith(client)),
    ).rejects.toBeInstanceOf(GalaxyValidationError);
    await expect(getJobDetails({}, ctxWith(client))).rejects.toBeInstanceOf(GalaxyValidationError);
    await expect(
      getJobDetails({ datasetId: null, jobId: null }, ctxWith(client)),
    ).rejects.toThrow("neither was given");
    expect(calls).toBe(0);
  });

  it("mirrors the Python signature in its schema", () => {
    const schema = z.object(getJobDetailsOp.input);
    expect(schema.parse({})).toEqual({ full: false });
    expect(schema.parse({ datasetId: null, jobId: null, historyId: null })).toEqual({
      datasetId: null,
      jobId: null,
      historyId: null,
      full: false,
    });
    expect(schema.safeParse({ full: null }).success).toBe(false);
    expect(schema.safeParse({ full: "yes" }).success).toBe(false);
  });
});
