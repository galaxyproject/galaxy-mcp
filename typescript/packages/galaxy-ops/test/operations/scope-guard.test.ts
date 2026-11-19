/**
 * The scope guard, through the registry: what a scoped context refuses, narrows and lets by,
 * and that a context without one behaves as every context did before.
 *
 * Every op here is called the way a surface or a library caller calls it -- `runWithEnvelope`
 * or the op's own function -- so what is tested is the guard the registry installed around
 * `run`, not a helper called by hand.
 */
import { describe, it, expect } from "vitest";
import "../../src/operations/all";
import { DEFAULT_POLL, type GalaxyContext } from "../../src/context";
import { GalaxyHandleKindError, GalaxyScopeError } from "../../src/errors";
import { HandleRegistry, type OperationScope } from "../../src/scope";
import { runWithEnvelope } from "../../src/operations/registry";
import { cancelWorkflowInvocationOp } from "../../src/operations/cancel-workflow-invocation";
import { createHistoryOp } from "../../src/operations/create-history";
import { createPageOp } from "../../src/operations/create-page";
import { downloadDatasetOp } from "../../src/operations/download-dataset";
import { getCollectionDetailsOp } from "../../src/operations/get-collection-details";
import { getDatasetDetailsOp, getDatasetDetails } from "../../src/operations/get-dataset-details";
import { getHistoriesOp } from "../../src/operations/get-histories";
import { getHistoryContentsOp } from "../../src/operations/get-history-contents";
import { getHistoryDetailsOp, getHistoryDetails } from "../../src/operations/get-history-details";
import { getInvocationsOp } from "../../src/operations/get-invocations";
import { getJobDetailsOp } from "../../src/operations/get-job-details";
import { getJobLogsOp } from "../../src/operations/get-job-logs";
import { getUserOp } from "../../src/operations/get-user";
import { getServerInfoOp } from "../../src/operations/get-server-info";
import { invokeWorkflowOp } from "../../src/operations/invoke-workflow";
import { listHistoryIdsOp } from "../../src/operations/list-history-ids";
import { listJobsOp, listJobs } from "../../src/operations/list-jobs";
import { runToolOp } from "../../src/operations/run-tool";
import { updateHistoryOp } from "../../src/operations/update-history";
import { uploadFileFromUrlOp } from "../../src/operations/upload-file-from-url";
import { mockClient } from "../util/mock-client";

const SCOPE: OperationScope = { historyIds: ["h1"], readOnlyHistoryIds: ["r1"] };

/** A context the way a host builds one for a run: scope, handle memory, nothing else. */
const scoped = (client: unknown, scope: OperationScope = SCOPE, handles?: Map<string, Set<never>> | null): GalaxyContext => ({
  client: client as GalaxyContext["client"],
  poll: DEFAULT_POLL,
  scope,
  createdHistories: new Set<string>(),
  ...(handles === null ? {} : { handles: (handles ?? new HandleRegistry()) as GalaxyContext["handles"] }),
});

/** An unscoped context, which is what every caller had before scopes existed. */
const unscoped = (client: unknown): GalaxyContext => ({ client: client as GalaxyContext["client"], poll: DEFAULT_POLL });

/** A client whose every request is a test failure: the call under test must be refused first. */
function neverClient() {
  const refuse = (method: string) => (path: string) => {
    throw new Error(`no request expected, but ${method} ${path} was made`);
  };
  return mockClient({ GET: refuse("GET"), POST: refuse("POST"), PUT: refuse("PUT"), DELETE: refuse("DELETE") });
}

/** A client that answers every route from a table and counts what it was asked. */
function tableClient(routes: Record<string, unknown | ((init: any) => unknown)>) {
  const asked: string[] = [];
  const answer = (method: string) => (path: string, init?: any) => {
    asked.push(`${method} ${path}`);
    const route = routes[`${method} ${path}`] ?? routes[path];
    if (route === undefined) throw new Error(`unexpected ${method} ${path}`);
    const data = typeof route === "function" ? route(init) : route;
    return { data, response: { status: 200 } };
  };
  return { asked, client: mockClient({ GET: answer("GET"), POST: answer("POST"), PUT: answer("PUT"), DELETE: answer("DELETE") }) };
}

const call = (op: unknown, input: unknown, ctx: GalaxyContext) => runWithEnvelope(op as never, input as never, ctx);

const HISTORIES = [
  { id: "h1", name: "mine" },
  { id: "h2", name: "theirs" },
  { id: "r1", name: "shared" },
];

describe("a history argument", () => {
  it("passes when the history is in scope, and the request goes out", async () => {
    const { asked, client } = tableClient({
      "/api/histories/{history_id}": { id: "h1", name: "mine" },
      "/api/histories/{history_id}/contents": [],
    });
    const out = await getHistoryDetails({ historyId: "h1" }, scoped(client));
    expect(out).toEqual({ id: "h1", name: "mine" });
    expect(asked).toEqual(["GET /api/histories/{history_id}", "GET /api/histories/{history_id}/contents"]);
  });

  it("is refused before any request when the history is outside the scope", async () => {
    const result = await call(getHistoryDetailsOp, { historyId: "h2" }, scoped(neverClient()));
    expect(result.success).toBe(false);
    expect(result.errorKind).toBe("out_of_scope");
    expect(result.message).toBe(
      "history_id 'h2' is outside this run's scope. Histories this run may read: h1, r1. Nothing was sent to Galaxy.",
    );
    expect(result.data).toBeUndefined();
    // And as a typed error for a library caller.
    await expect(getHistoryDetails({ historyId: "h2" }, scoped(neverClient()))).rejects.toBeInstanceOf(GalaxyScopeError);
  });

  it("compares ids the way Galaxy does, without regard to case", async () => {
    const { client } = tableClient({
      "/api/histories/{history_id}": { id: "ABC123" },
      "/api/histories/{history_id}/contents": [],
    });
    const result = await call(getHistoryDetailsOp, { historyId: "ABC123" }, scoped(client, { historyIds: ["abc123"] }));
    expect(result.success).toBe(true);
  });

  it("lets a read-only history be read but not changed", async () => {
    const { client } = tableClient({
      "/api/histories/{history_id}/contents": { contents: [], stats: { total_matches: 0 } },
    });
    const read = await call(getHistoryContentsOp, { historyId: "r1", limit: 10, offset: 0 }, scoped(client));
    expect(read.success).toBe(true);

    const write = await call(updateHistoryOp, { historyId: "r1", name: "renamed" }, scoped(neverClient()));
    expect(write.errorKind).toBe("out_of_scope");
    expect(write.message).toBe(
      "history_id 'r1' is read-only in this run and update_history changes it. " +
        "Histories this run may write: h1; read-only: r1. Nothing was sent to Galaxy.",
    );
  });

  it("is checked on the optional filters too: list_jobs for another history is refused", async () => {
    const result = await call(listJobsOp, { historyId: "h2", limit: 10, offset: 0 }, scoped(neverClient()));
    expect(result.errorKind).toBe("out_of_scope");
  });
});

describe("a write that would land in Galaxy's current history", () => {
  it("is refused under a scope when the history is left out, and runs when it is named", async () => {
    const missing = await call(uploadFileFromUrlOp, { url: "https://x.example/a.txt", fileType: "auto", dbkey: "?" }, scoped(neverClient()));
    expect(missing.errorKind).toBe("out_of_scope");
    expect(missing.message).toBe(
      "upload_file_from_url needs history_id in this run: without it Galaxy would use its current " +
        "history, which is not known to be one this run may write. Histories this run may write: h1; " +
        "read-only: r1. Nothing was sent to Galaxy.",
    );
    const { asked, client } = tableClient({ "POST /api/tools": { outputs: [{ id: "d1" }], jobs: [{ id: "j1" }] } });
    const named = await call(uploadFileFromUrlOp, { url: "https://x.example/a.txt", historyId: "h1", fileType: "auto", dbkey: "?" }, scoped(client));
    expect(named.success).toBe(true);
    expect(asked).toEqual(["POST /api/tools"]);
  });

  it("holds invoke_workflow to a history id or a name for a new one, and a new one to the right to create", async () => {
    const neither = await call(
      invokeWorkflowOp,
      { workflowId: "wf1", inputsBy: "step_index", parametersNormalized: false },
      scoped(neverClient()),
    );
    expect(neither.errorKind).toBe("out_of_scope");
    expect(neither.message).toMatch(/^invoke_workflow needs history_id or history_name in this run/);

    const noCreate = await call(
      invokeWorkflowOp,
      { workflowId: "wf1", historyName: "fresh", inputsBy: "step_index", parametersNormalized: false },
      scoped(neverClient(), { ...SCOPE, allowCreate: false }),
    );
    expect(noCreate.message).toBe(
      "invoke_workflow would create a history, and this run may not create histories. Nothing was sent to Galaxy.",
    );

    // By name, the history Galaxy made for the run joins the session's histories.
    const { client } = tableClient({
      "POST /api/workflows/{workflow_id}/invocations": { id: "i1", history_id: "d4", workflow_id: "wf1" },
      "/api/histories/{history_id}": { id: "d4" },
      "/api/histories/{history_id}/contents": [],
    });
    const ctx = scoped(client);
    const named = await call(
      invokeWorkflowOp,
      { workflowId: "wf1", historyName: "fresh", inputsBy: "step_index", parametersNormalized: false },
      ctx,
    );
    expect(named.success).toBe(true);
    expect(ctx.createdHistories).toEqual(new Set(["d4"]));
    expect((await call(getHistoryDetailsOp, { historyId: "d4" }, ctx)).success).toBe(true);
  });

  it("does not apply to a page, whose absent history means a standalone report", async () => {
    const { client } = tableClient({ "POST /api/pages": { id: "p1", title: "Report" } });
    const ctx = { ...scoped(client), galaxyVersion: async () => ({ source: "unknown" as const }) };
    const result = await call(createPageOp, { title: "Report", slug: "report" }, ctx);
    expect(result.success).toBe(true);
  });
});

describe("a record read by a non-history id", () => {
  const dataset = (history_id: string) => ({ id: "d1", name: "reads", state: "ok", history_id, creating_job: "j1" });

  it("is refused after the read, and not returned, when it belongs to another history", async () => {
    const { asked, client } = tableClient({
      "/api/datasets/{dataset_id}": dataset("h2"),
      "/api/datasets/{dataset_id}/get_content_as_text": { item_data: "ACGT\n", truncated: false },
    });
    const result = await call(getDatasetDetailsOp, { datasetId: "d1", includePreview: true, previewLines: 10 }, scoped(client));
    expect(result.success).toBe(false);
    expect(result.errorKind).toBe("out_of_scope");
    expect(result.data).toBeUndefined();
    expect(result.message).toBe(
      "dataset 'd1' belongs to history 'h2', which is outside this run's scope. " +
        "Histories this run may read: h1, r1. It was not returned.",
    );
    expect(JSON.stringify(result)).not.toContain("ACGT");
    expect(asked[0]).toBe("GET /api/datasets/{dataset_id}");
  });

  it("is returned when it belongs to a scoped history, read-only included", async () => {
    const { client } = tableClient({
      "/api/datasets/{dataset_id}": dataset("r1"),
      "/api/datasets/{dataset_id}/get_content_as_text": { item_data: "ACGT\n", truncated: false },
    });
    const result = await call(getDatasetDetailsOp, { datasetId: "d1", includePreview: true, previewLines: 10 }, scoped(client));
    expect(result.success).toBe(true);
    expect((result.data as any).dataset.history_id).toBe("r1");
  });

  it("holds a collection, a job and one invocation to the same rule", async () => {
    const outside = tableClient({
      "/api/dataset_collections/{hdca_id}": { id: "c1", history_id: "h2", elements: [] },
      "/api/jobs/{job_id}": { id: "a1b2", history_id: "h2", state: "ok" },
      "/api/invocations/{invocation_id}": { id: "a1b3", history_id: "h2", state: "scheduled" },
      "/api/invocations/{invocation_id}/jobs_summary": { states: {} },
    });
    for (const [op, input] of [
      [getCollectionDetailsOp, { collectionId: "c1", maxElements: 100 }],
      [getJobDetailsOp, { jobId: "a1b2" }],
      [getInvocationsOp, { invocationId: "a1b3", view: "collection", stepDetails: false, offset: 0, includeTerminal: true }],
    ] as const) {
      const result = await call(op, input, scoped(outside.client));
      expect(result.errorKind, op.name).toBe("out_of_scope");
      expect(result.message, op.name).toContain("belongs to history 'h2', which is outside this run's scope");
      expect(result.data, op.name).toBeUndefined();
    }
  });

  it("is looked up BEFORE a cancel, so the DELETE of another history's invocation is never sent", async () => {
    const { asked, client } = tableClient({
      "GET /api/invocations/{invocation_id}": { id: "a1b3", history_id: "h2" },
      "DELETE /api/invocations/{invocation_id}": { id: "a1b3", state: "cancelled" },
    });
    const result = await call(cancelWorkflowInvocationOp, { invocationId: "a1b3" }, scoped(client));
    expect(result.errorKind).toBe("out_of_scope");
    expect(result.message).toBe(
      "invocation 'a1b3' belongs to history 'h2', which is outside this run's scope. " +
        "Histories this run may write: h1; read-only: r1. cancel_workflow_invocation was not run.",
    );
    expect(asked).toEqual(["GET /api/invocations/{invocation_id}"]);

    const own = tableClient({
      "GET /api/invocations/{invocation_id}": { id: "a1b3", history_id: "h1" },
      "DELETE /api/invocations/{invocation_id}": { id: "a1b3", history_id: "h1", state: "cancelled" },
    });
    expect((await call(cancelWorkflowInvocationOp, { invocationId: "a1b3" }, scoped(own.client))).success).toBe(true);
    expect(own.asked).toEqual(["GET /api/invocations/{invocation_id}", "DELETE /api/invocations/{invocation_id}"]);
  });

  it("is looked up BEFORE a download, and a read-only history may serve one", async () => {
    const meta = (history_id: string) => ({ id: "d1", name: "reads", file_ext: "txt", state: "ok", history_id, file_size: 3 });
    const outside = tableClient({
      "/api/datasets/{dataset_id}": meta("h2"),
      "/api/datasets/{dataset_id}/display": new Uint8Array([1, 2, 3]).buffer,
    });
    const refused = await call(downloadDatasetOp, { datasetId: "d1", useDefaultFilename: true, requireOkState: true }, scoped(outside.client));
    expect(refused.errorKind).toBe("out_of_scope");
    expect(refused.message).toContain("download_dataset was not run.");
    expect(outside.asked).toEqual(["GET /api/datasets/{dataset_id}"]);

    const readOnly = tableClient({
      "/api/datasets/{dataset_id}": meta("r1"),
      "/api/datasets/{dataset_id}/display": new Uint8Array([1, 2, 3]).buffer,
    });
    const served = await call(downloadDatasetOp, { datasetId: "d1", useDefaultFilename: true, requireOkState: true }, scoped(readOnly.client));
    expect(served.success).toBe(true);
    expect(readOnly.asked.at(-1)).toBe("GET /api/datasets/{dataset_id}/display");
  });

  it("reads the job before its logs, whose answer carries no history of its own", async () => {
    const { asked, client } = tableClient({
      "/api/jobs/{job_id}": (init: any) =>
        init?.params?.query?.full ? { id: "a1b2", history_id: "h2", stdout: "secret", stderr: "" } : { id: "a1b2", history_id: "h2" },
    });
    const result = await call(getJobLogsOp, { jobId: "a1b2", logBytes: 4096 }, scoped(client));
    expect(result.errorKind).toBe("out_of_scope");
    expect(JSON.stringify(result)).not.toContain("secret");
    expect(asked).toEqual(["GET /api/jobs/{job_id}"]);
  });

  it("lets the op run and fail in its own words when the owner lookup itself fails", async () => {
    let gets = 0;
    const client = mockClient({
      GET: () => {
        gets += 1;
        return { error: { err_msg: "boom" }, response: { status: 500 } };
      },
      DELETE: () => ({ error: { err_msg: "no such invocation" }, response: { status: 404 } }),
    });
    const result = await call(cancelWorkflowInvocationOp, { invocationId: "a1b3" }, scoped(client));
    expect(gets).toBe(1);
    expect(result.errorKind).toBe("not_found");
  });
});

describe("a listing", () => {
  it("of histories is narrowed to the scope, with the count and the page describing what is left", async () => {
    const { client } = tableClient({ "/api/histories": HISTORIES });
    const result = await call(getHistoriesOp, { offset: 0 }, scoped(client));
    expect(result.success).toBe(true);
    expect(result.data).toEqual([{ id: "h1", name: "mine" }, { id: "r1", name: "shared" }]);
    expect(result.count).toBe(2);
    expect(result.message).toBe("Retrieved 2 histories");

    const ids = await call(listHistoryIdsOp, { limit: 100, offset: 0 }, scoped(client));
    expect(ids.data).toEqual([{ id: "h1", name: "mine" }, { id: "r1", name: "shared" }]);
    expect(ids.pagination?.total_items).toBe(2);
    expect(ids.message).toBe("Found 2 of 2 histories");
  });

  it("of histories is narrowed before it is paged, so a window and its total describe the scope", async () => {
    const { client } = tableClient({ "/api/histories": HISTORIES });
    const result = await call(getHistoriesOp, { limit: 1, offset: 1 }, scoped(client));
    expect(result.data).toEqual([{ id: "r1", name: "shared" }]);
    expect(result.pagination?.total_items).toBe(2);
    expect(result.pagination?.has_next).toBe(false);
    expect(result.message).toBe("Retrieved 1 of 2 histories");
  });

  it("of jobs and of invocations is narrowed by each row's history", async () => {
    const { client } = tableClient({
      "/api/jobs": [
        { id: "j1", history_id: "h1", state: "ok" },
        { id: "j2", history_id: "h2", state: "ok" },
        { id: "j3", history_id: null, state: "ok" },
      ],
      "/api/invocations": [
        { id: "i1", history_id: "h1" },
        { id: "i2", history_id: "h2" },
        { id: "i3", history_id: "r1" },
      ],
    });
    const jobs = await call(listJobsOp, { orderBy: "update_time", view: "collection", limit: 100, offset: 0 }, scoped(client));
    expect((jobs.data as any[]).map((j) => j.id)).toEqual(["j1"]);
    expect(jobs.count).toBe(1);
    const invocations = await call(
      getInvocationsOp,
      { view: "collection", stepDetails: false, offset: 0, includeTerminal: true },
      scoped(client),
    );
    expect((invocations.data as any[]).map((i) => i.id)).toEqual(["i1", "i3"]);
    expect(invocations.count).toBe(2);
  });
});

describe("a blocked operation", () => {
  it("is refused by name before any request", async () => {
    const ctx = scoped(neverClient(), { ...SCOPE, blockedOperations: ["create_history"] });
    const result = await call(createHistoryOp, { historyName: "new" }, ctx);
    expect(result.errorKind).toBe("out_of_scope");
    expect(result.message).toBe(
      "create_history is not available in this run: the host's scope blocks it. Nothing was sent to Galaxy.",
    );
  });

  it("is only the ones named: create_history runs under a scope that does not block it", async () => {
    const { client } = tableClient({ "POST /api/histories": { id: "h9", name: "new" } });
    const result = await call(createHistoryOp, { historyName: "new" }, scoped(client));
    expect(result.success).toBe(true);
  });
});

describe("an id passed as a kind it was not returned as", () => {
  it("is refused before any request, naming both kinds", async () => {
    const handles = new HandleRegistry();
    const { client } = tableClient({ "/api/jobs": [{ id: "a1b2", history_id: "h1", state: "ok" }] });
    await listJobs({ historyId: "h1" }, scoped(client, SCOPE, handles));
    expect(handles.get("a1b2")).toEqual(new Set(["job"]));

    const ctx = scoped(neverClient(), SCOPE, handles);
    const result = await call(getDatasetDetailsOp, { datasetId: "a1b2", includePreview: true, previewLines: 10 }, ctx);
    expect(result.success).toBe(false);
    expect(result.errorKind).toBe("handle_kind_mismatch");
    expect(result.message).toBe(
      "dataset_id 'a1b2' was returned by an earlier call as a job id; dataset_id expects a dataset id. " +
        "Nothing was sent to Galaxy.",
    );
    await expect(getDatasetDetails({ datasetId: "a1b2" }, ctx)).rejects.toBeInstanceOf(GalaxyHandleKindError);
  });

  it("passes when the id was also returned as the expected kind", async () => {
    const handles = new HandleRegistry();
    handles.set("a1b2", new Set(["job", "hda"]));
    const { asked, client } = tableClient({ "/api/datasets/{dataset_id}": { id: "a1b2", history_id: "h1", state: "queued" } });
    const result = await call(getDatasetDetailsOp, { datasetId: "a1b2", includePreview: true, previewLines: 10 }, scoped(client, SCOPE, handles));
    expect(result.success).toBe(true);
    expect(asked).toEqual(["GET /api/datasets/{dataset_id}"]);
  });

  it("is a check a host can have without a scope, by supplying the handle memory", async () => {
    const handles = new HandleRegistry();
    handles.set("a1b2", new Set(["job"]));
    const ctx: GalaxyContext = { ...unscoped(neverClient()), handles };
    await expect(getDatasetDetails({ datasetId: "a1b2" }, ctx)).rejects.toBeInstanceOf(GalaxyHandleKindError);
  });

  it("is not a check at all for a scoped context whose host turned the memory off", async () => {
    const { asked, client } = tableClient({ "/api/datasets/{dataset_id}": { id: "a1b2", history_id: "h1", state: "queued" } });
    const result = await call(getDatasetDetailsOp, { datasetId: "a1b2", includePreview: true, previewLines: 10 }, scoped(client, SCOPE, null));
    expect(result.success).toBe(true);
    expect(asked.length).toBe(1);
  });
});

describe("an id the run has never seen", () => {
  it("passes the kind check, because Galaxy's ids carry no type", async () => {
    const handles = new HandleRegistry();
    handles.set("other", new Set(["job"]));
    const { asked, client } = tableClient({ "/api/datasets/{dataset_id}": { id: "zzz", history_id: "h1", state: "queued" } });
    const result = await call(getDatasetDetailsOp, { datasetId: "zzz", includePreview: true, previewLines: 10 }, scoped(client, SCOPE, handles));
    expect(result.success).toBe(true);
    expect(asked).toEqual(["GET /api/datasets/{dataset_id}"]);
    // And what the read returned is remembered for next time.
    expect(handles.get("zzz")).toEqual(new Set(["hda"]));
    expect(handles.get("h1")).toEqual(new Set(["history"]));
  });
});

describe("what the ops remember", () => {
  it("records the datasets and collections a history lists, by kind", async () => {
    const handles = new HandleRegistry();
    const { client } = tableClient({
      "/api/histories/{history_id}/contents": {
        contents: [
          { id: "d1", history_content_type: "dataset" },
          { id: "c1", history_content_type: "dataset_collection" },
          { id: "c2", collection_type: "list" },
        ],
        stats: { total_matches: 3 },
      },
    });
    await call(getHistoryContentsOp, { historyId: "h1", limit: 10, offset: 0 }, scoped(client, SCOPE, handles));
    expect(handles.get("d1")).toEqual(new Set(["hda"]));
    expect(handles.get("c1")).toEqual(new Set(["hdca"]));
    expect(handles.get("c2")).toEqual(new Set(["hdca"]));
  });

  it("records a collection's elements apart from the datasets behind them", async () => {
    const handles = new HandleRegistry();
    const { client } = tableClient({
      "/api/dataset_collections/{hdca_id}": {
        id: "c1",
        history_id: "h1",
        elements: [
          { id: "e1", element_type: "hda", object: { id: "d1" } },
          { id: "e2", element_type: "dataset_collection", object: { id: "c9" } },
        ],
      },
    });
    await call(getCollectionDetailsOp, { collectionId: "c1", maxElements: 100 }, scoped(client, SCOPE, handles));
    expect(handles.get("c1")).toEqual(new Set(["hdca"]));
    expect(handles.get("e1")).toEqual(new Set(["dataset_collection_element"]));
    expect(handles.get("d1")).toEqual(new Set(["hda"]));
    expect(handles.get("c9")).toEqual(new Set(["hdca"]));
    // So an element id handed to get_dataset_details is caught.
    const result = await call(getDatasetDetailsOp, { datasetId: "e1", includePreview: true, previewLines: 10 }, scoped(neverClient(), SCOPE, handles));
    expect(result.errorKind).toBe("handle_kind_mismatch");
    expect(result.message).toContain("as a collection element id; dataset_id expects a dataset id");
  });

  it("records what a tool run and a job record name", async () => {
    // Hex ids throughout: a read by job id refuses anything else before asking Galaxy.
    const handles = new HandleRegistry();
    const { client } = tableClient({
      "/api/tools/{tool_id}/build": {},
      "/api/tools/{tool_id}": {},
      "POST /api/tools": {
        jobs: [{ id: "a1", tool_version: "1.0" }],
        outputs: [{ id: "d1" }],
        output_collections: [{ id: "c1" }],
        implicit_collections: [{ id: "c2" }],
      },
      "/api/jobs/{job_id}": {
        id: "a1",
        history_id: "h1",
        inputs: { input1: { src: "hda", id: "d0" } },
        outputs: { out1: { src: "hda", id: "d1" }, out2: { src: "hdca", id: "c1" } },
      },
    });
    const ctx = scoped(client, SCOPE, handles);
    const run = await call(runToolOp, { toolId: "cat1", historyId: "h1", inputs: {} }, ctx);
    expect(run.success).toBe(true);
    expect(handles.get("a1")).toEqual(new Set(["job"]));
    expect(handles.get("d1")).toEqual(new Set(["hda"]));
    expect(handles.get("c1")).toEqual(new Set(["hdca"]));
    expect(handles.get("c2")).toEqual(new Set(["hdca"]));
    const job = await call(getJobDetailsOp, { jobId: "a1" }, ctx);
    expect(job.success).toBe(true);
    expect(handles.get("d0")).toEqual(new Set(["hda"]));
    expect(handles.get("h1")).toEqual(new Set(["history"]));
  });
});

describe("the account surface under a scope", () => {
  const record = {
    id: "u1",
    email: "alice@example.org",
    username: "alice",
    preferences: { theme: "dark", "extra_user_preferences": "{}" },
    api_key: "SECRET-KEY-VALUE",
    total_disk_usage: 1048576,
    quota_percent: 12.5,
  };

  it("is the id and the username, and nothing else", async () => {
    const { client } = tableClient({ "/api/users/{user_id}": record });
    const result = await call(getUserOp, {}, scoped(client));
    expect(result.success).toBe(true);
    expect(result.data).toEqual({ id: "u1", username: "alice" });
    expect(result.message).toBe("Retrieved user info for 'alice'");
    const text = JSON.stringify(result);
    for (const forbidden of ["email", "preferences", "api_key", "SECRET-KEY-VALUE", "quota", "disk_usage"]) {
      expect(text).not.toContain(forbidden);
    }
  });

  it("is the whole record without a scope, as it always was", async () => {
    const { client } = tableClient({ "/api/users/{user_id}": record });
    const result = await call(getUserOp, {}, unscoped(client));
    expect(result.data).toEqual(record);
  });
});

describe("a context without a scope", () => {
  it("returns a record from any history, every history, every job, and remembers nothing", async () => {
    const { client } = tableClient({
      "/api/datasets/{dataset_id}": { id: "d1", history_id: "h2", state: "queued" },
      "/api/histories": HISTORIES,
      "/api/jobs": [{ id: "j1", history_id: "h2" }],
    });
    const ctx = unscoped(client);
    expect((await call(getDatasetDetailsOp, { datasetId: "d1", includePreview: true, previewLines: 10 }, ctx)).success).toBe(true);
    expect(((await call(getHistoriesOp, { offset: 0 }, ctx)).data as unknown[]).length).toBe(3);
    expect(((await call(listJobsOp, { orderBy: "update_time", view: "collection", limit: 100, offset: 0 }, ctx)).data as unknown[]).length).toBe(1);
    expect(ctx.handles).toBeUndefined();
  });

  it("runs an op a scope would block, and a write with no history named", async () => {
    const { client } = tableClient({
      "POST /api/histories": { id: "h9" },
      "POST /api/tools": { outputs: [], jobs: [] },
    });
    const ctx = unscoped(client);
    expect((await call(createHistoryOp, { historyName: "new" }, ctx)).success).toBe(true);
    expect((await call(uploadFileFromUrlOp, { url: "https://x.example/a", fileType: "auto", dbkey: "?" }, ctx)).success).toBe(true);
  });
});

describe("a history the session creates", () => {
  const routes = {
    "POST /api/histories": { id: "d4", name: "scratch" },
    "/api/histories": [...HISTORIES, { id: "d4", name: "scratch" }],
    "POST /api/tools": { jobs: [{ id: "a9" }], outputs: [{ id: "d9" }] },
    "PUT /api/histories/{history_id}": (init: any) => ({ id: init.params.path.history_id, deleted: true }),
  };

  it("joins the histories it may write, read and list", async () => {
    const { client } = tableClient(routes);
    const ctx = scoped(client);
    const before = await call(runToolOp, { toolId: "cat1", historyId: "d4", inputs: {} }, ctx);
    expect(before.errorKind).toBe("out_of_scope");
    expect((await call(createHistoryOp, { historyName: "scratch" }, ctx)).success).toBe(true);
    expect(ctx.createdHistories).toEqual(new Set(["d4"]));
    expect((await call(runToolOp, { toolId: "cat1", historyId: "d4", inputs: {} }, ctx)).success).toBe(true);
    const listed = await call(getHistoriesOp, { offset: 0 }, ctx);
    expect((listed.data as any[]).map((h) => h.id)).toEqual(["h1", "r1", "d4"]);
  });

  it("does not join them when the host said includeCreated: false", async () => {
    const { client } = tableClient(routes);
    const ctx = scoped(client, { ...SCOPE, includeCreated: false });
    expect((await call(createHistoryOp, { historyName: "scratch" }, ctx)).success).toBe(true);
    expect((await call(runToolOp, { toolId: "cat1", historyId: "d4", inputs: {} }, ctx)).errorKind).toBe("out_of_scope");
  });

  it("cannot be created when the host turned creating off", async () => {
    const result = await call(createHistoryOp, { historyName: "scratch" }, scoped(neverClient(), { ...SCOPE, allowCreate: false }));
    expect(result.errorKind).toBe("out_of_scope");
    expect(result.message).toBe(
      "create_history would create a history, and this run may not create histories. Nothing was sent to Galaxy.",
    );
  });

  it("is the only kind it may delete by default; a writable one it did not create is refused", async () => {
    const { asked, client } = tableClient(routes);
    const ctx = scoped(client);
    await call(createHistoryOp, { historyName: "scratch" }, ctx);
    const own = await call(updateHistoryOp, { historyId: "h1", deleted: true }, ctx);
    expect(own.errorKind).toBe("out_of_scope");
    expect(own.message).toBe(
      "update_history would delete history_id 'h1', which this run may not delete. " +
        "This run may delete only histories it created itself: d4. Nothing was sent to Galaxy.",
    );
    expect(asked.filter((a) => a.startsWith("PUT"))).toEqual([]);
    // Renaming it is a write, not a delete, and restoring is not a delete either.
    expect((await call(updateHistoryOp, { historyId: "h1", name: "renamed" }, ctx)).success).toBe(true);
    expect((await call(updateHistoryOp, { historyId: "h1", deleted: false }, ctx)).success).toBe(true);
    expect((await call(updateHistoryOp, { historyId: "d4", deleted: true }, ctx)).success).toBe(true);
  });

  it("is not the only kind when the host set destructive: writable -- but read-only stays read-only", async () => {
    const { client } = tableClient(routes);
    const ctx = scoped(client, { ...SCOPE, destructive: "writable" });
    expect((await call(updateHistoryOp, { historyId: "h1", deleted: true }, ctx)).success).toBe(true);
    expect((await call(updateHistoryOp, { historyId: "r1", deleted: true }, ctx)).errorKind).toBe("out_of_scope");
  });
});

describe("get_server_info under a scope", () => {
  const routes = { "/api/configuration": { brand: "Test" } };
  const versioned = (ctx: GalaxyContext): GalaxyContext => ({
    ...ctx,
    baseUrl: "https://g.example",
    galaxyVersion: async () => ({ source: "unknown" as const }),
  });

  it("tells the session what it may touch, in data and in one sentence", async () => {
    const { client } = tableClient(routes);
    const result = await call(getServerInfoOp, {}, versioned(scoped(client)));
    expect((result.data as any).scope).toEqual({
      readable_history_ids: ["h1", "r1"],
      writable_history_ids: ["h1"],
      created_history_ids: [],
      create_allowed: true,
      destructive: "created",
      blocked_operations: [],
      note:
        "This session is scoped. It may read histories: h1, r1; write to: h1. " +
        "It may create histories, and use the ones it creates. It may delete only histories it created itself.",
    });
    expect(result.message).toBe(
      "Retrieved server info for https://g.example. This session is scoped. It may read histories: h1, r1; " +
        "write to: h1. It may create histories, and use the ones it creates. " +
        "It may delete only histories it created itself.",
    );
  });

  it("says nothing about a scope when there is none, so the unscoped answer is what it was", async () => {
    const { client } = tableClient(routes);
    const result = await call(getServerInfoOp, {}, versioned(unscoped(client)));
    expect("scope" in (result.data as object)).toBe(false);
    expect(result.message).toBe("Retrieved server info for https://g.example");
  });
});
