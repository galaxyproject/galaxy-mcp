/**
 * The payload is Galaxy's legacy workflow-creation form (webapps/galaxy/api/workflows.py,
 * `create`, the `from_history_id` branch): `job_ids` are encoded job ids Galaxy decodes
 * itself, `dataset_ids` and `dataset_collection_ids` are HIDS -- history item numbers,
 * read with `int()` in workflow/extract.py -- and `workflow_name` is required.
 */
import { describe, it, expect } from "vitest";
import {
  extractWorkflowFromHistoryOp,
  extractWorkflowFromHistory,
  extractWorkflowPayload,
} from "../../src/operations/extract-workflow-from-history";
import { GalaxyConnectionError, GalaxyValidationError, GalaxyAuthError } from "../../src/errors";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

/** What Galaxy answers: StoredWorkflow.to_dict("collection") with the id encoded, plus a url. */
const STORED = {
  model_class: "StoredWorkflow",
  id: "wf42",
  latest_workflow_id: "lw7",
  name: "Per-sample QC",
  create_time: "2026-10-31T10:00:00",
  update_time: "2026-10-31T10:00:00",
  published: false,
  importable: false,
  deleted: false,
  hidden: false,
  latest_workflow_uuid: "0d1c7b9c-5a19-4a8e-9f8e-2d4c6c6f3f11",
  creator_deleted: false,
  url: "/api/workflows/wf42",
};

describe("extract_workflow_from_history", () => {
  it("is a write op", () => {
    expect(extractWorkflowFromHistoryOp.readOnly).toBe(false);
    expect(extractWorkflowFromHistoryOp.domain).toBe("workflows");
  });

  it("POSTs Galaxy's legacy form: encoded job ids, hids as integers, the history and the name", async () => {
    let captured: { path: string; body: unknown } | undefined;
    const client = mockClient({
      POST: (path, init) => {
        captured = { path, body: init?.body };
        return { data: STORED, response: { status: 200 } };
      },
    });
    const out = await extractWorkflowFromHistory(
      {
        historyId: "h1",
        workflowName: "Per-sample QC",
        jobIds: ["j1", "j2"],
        datasetHids: [3, "4"],
        collectionHids: ["12"],
      },
      ctxWith(client),
    );
    expect(captured?.path).toBe("/api/workflows");
    expect(captured?.body).toEqual({
      from_history_id: "h1",
      workflow_name: "Per-sample QC",
      job_ids: ["j1", "j2"],
      dataset_ids: [3, 4],
      dataset_collection_ids: [12],
    });
    // Nothing else goes in the body: Galaxy refuses a payload naming two ways to create.
    expect(Object.keys(captured?.body as object).sort()).toEqual([
      "dataset_collection_ids",
      "dataset_ids",
      "from_history_id",
      "job_ids",
      "workflow_name",
    ]);

    expect(out.workflow_id).toBe("wf42");
    expect(out.name).toBe("Per-sample QC");
    // StoredWorkflow.to_dict carries no step count, so none is invented.
    expect(out.number_of_steps).toBeNull();
    expect(out.steps_requested).toEqual({ tool_steps: 2, dataset_inputs: 2, collection_inputs: 1 });
    expect(out.history_id).toBe("h1");
    expect(out.job_ids).toEqual(["j1", "j2"]);
    expect(out.dataset_hids).toEqual([3, 4]);
    expect(out.collection_hids).toEqual([12]);
    expect(out.workflow).toEqual(STORED);
    expect(out.next_steps).toBe(
      "To run it on other inputs, call get_workflow_input_template with workflow_id 'wf42' to see its " +
        "input slots, fill them with the new datasets or collections, and invoke_workflow -- once for " +
        "every set of inputs, instead of re-running the tools one by one.",
    );
  });

  it("reports a step count when Galaxy's reply has one", async () => {
    const client = mockClient({
      POST: () => ({ data: { ...STORED, number_of_steps: 3 }, response: { status: 200 } }),
    });
    const out = await extractWorkflowFromHistory(
      { historyId: "h1", workflowName: "x", jobIds: ["j1"] },
      ctxWith(client),
    );
    expect(out.number_of_steps).toBe(3);
  });

  it("sends empty lists for what was not named", () => {
    expect(extractWorkflowPayload({ historyId: "h1", workflowName: " QC ", jobIds: ["j1"] })).toEqual({
      from_history_id: "h1",
      workflow_name: "QC",
      job_ids: ["j1"],
      dataset_ids: [],
      dataset_collection_ids: [],
    });
  });

  it("words the message with what was requested and where to go next", async () => {
    const client = mockClient({ POST: () => ({ data: STORED, response: { status: 200 } }) });
    const out = await extractWorkflowFromHistory(
      { historyId: "h1", workflowName: "Per-sample QC", jobIds: ["j1", "j2"], datasetHids: [3] },
      ctxWith(client),
    );
    const message = extractWorkflowFromHistoryOp.project!(out, {
      historyId: "h1",
      workflowName: "Per-sample QC",
    }).message;
    expect(message).toBe(
      "Extracted workflow 'Per-sample QC' (id wf42) from history h1: 2 tool step(s) from the listed " +
        "jobs, 1 dataset input(s) and 0 collection input(s). " +
        out.next_steps,
    );
  });

  describe("refuses before anything is sent", () => {
    const neverPosts = () =>
      mockClient({
        POST: () => {
          throw new Error("must not POST");
        },
      });

    it("an extraction that names no job and no input", async () => {
      await expect(
        extractWorkflowFromHistory({ historyId: "h1", workflowName: "empty" }, ctxWith(neverPosts())),
      ).rejects.toThrow(/Nothing to extract.*Nothing was sent to Galaxy/);
    });

    it("an empty workflow name", async () => {
      await expect(
        extractWorkflowFromHistory({ historyId: "h1", workflowName: "  ", jobIds: ["j1"] }, ctxWith(neverPosts())),
      ).rejects.toBeInstanceOf(GalaxyValidationError);
    });

    it("a hid that is not a history item number", async () => {
      // An encoded id where a hid belongs is the likely mistake, and Galaxy would answer it
      // with int()'s ValueError rather than an explanation.
      await expect(
        extractWorkflowFromHistory(
          { historyId: "h1", workflowName: "x", datasetHids: ["f2db41e1fa331b3e"] },
          ctxWith(neverPosts()),
        ),
      ).rejects.toThrow(/dataset_hids must be history item numbers.*got "f2db41e1fa331b3e"/);
      await expect(
        extractWorkflowFromHistory({ historyId: "h1", workflowName: "x", collectionHids: [0] }, ctxWith(neverPosts())),
      ).rejects.toBeInstanceOf(GalaxyValidationError);
      await expect(
        extractWorkflowFromHistory({ historyId: "h1", workflowName: "x", datasetHids: [2.5] }, ctxWith(neverPosts())),
      ).rejects.toBeInstanceOf(GalaxyValidationError);
    });
  });

  it("turns Galaxy's 400 into a validation failure that keeps the reply's facts", async () => {
    // workflow/extract.py: "hid 12 does not appear to be a collection" is a
    // RequestParameterInvalidException, which Galaxy answers as a 400 with err_code 400008.
    const client = mockClient({
      POST: () => ({
        error: { err_msg: "hid 12 does not appear to be a collection", err_code: 400008 },
        response: { status: 400 },
      }),
    });
    const failure = await extractWorkflowFromHistory(
      { historyId: "h1", workflowName: "x", jobIds: ["j1"], collectionHids: [12] },
      ctxWith(client),
    ).catch((e: unknown) => e);
    expect(failure).toBeInstanceOf(GalaxyValidationError);
    const err = failure as GalaxyValidationError;
    expect(err.kind).toBe("validation");
    expect(err.message).toBe("hid 12 does not appear to be a collection");
    expect(err.errCode).toBe(400008);
    expect(err.http?.status).toBe(400);
    expect(err.cause).toBeInstanceOf(GalaxyConnectionError);
  });

  it("leaves any other failure as it was classified", async () => {
    const client = mockClient({
      POST: () => ({ error: { err_msg: "Provided API key is not valid." }, response: { status: 403 } }),
    });
    await expect(
      extractWorkflowFromHistory({ historyId: "h1", workflowName: "x", jobIds: ["j1"] }, ctxWith(client)),
    ).rejects.toBeInstanceOf(GalaxyAuthError);
  });

  it("words a failed request as a bioblend write with the history and the name in the context", () => {
    const contract = extractWorkflowFromHistoryOp.failure!;
    expect(contract.shape).toBe("bioblend-write");
    expect(contract.action).toBe("Extract workflow from history");
    expect(contract.context!({ historyId: "h1", workflowName: "QC" })).toEqual({
      history_id: "h1",
      workflow_name: "QC",
    });
  });
});
