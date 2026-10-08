import { describe, it, expect, beforeEach, vi } from "vitest";
import { importWorkflowFromIwcOp, importWorkflowFromIwc } from "../../src/operations/import-workflow-from-iwc";
import { __resetIwcCacheForTest, __setIwcCacheForTest, type IwcWorkflow } from "../../src/iwc-manifest";
import { GalaxyNotFoundError } from "../../src/errors";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { mockClient } from "../util/mock-client";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const DEFINITION = {
  name: "RNA-Seq PE",
  annotation: "Paired-end RNA-seq",
  steps: {},
  format_version: "0.1",
};

const WF: IwcWorkflow = {
  trsID: "#workflow/github.com/iwc-workflows/rnaseq-pe/main",
  definition: DEFINITION,
  readme: "A great workflow",
};

beforeEach(() => {
  __resetIwcCacheForTest();
  vi.restoreAllMocks();
});

describe("import_workflow_from_iwc", () => {
  it("is a write op", () => {
    expect(importWorkflowFromIwcOp.readOnly).toBe(false);
  });

  it("throws GalaxyNotFoundError and does NOT POST when trsId is missing", async () => {
    __setIwcCacheForTest([WF]);
    let postCalled = false;
    const client = mockClient({
      POST: () => {
        postCalled = true;
        return { data: {}, response: { status: 200 } };
      },
    });
    await expect(importWorkflowFromIwc({ trsId: "nonexistent" }, ctxWith(client))).rejects.toBeInstanceOf(
      GalaxyNotFoundError,
    );
    expect(postCalled).toBe(false);
  });

  it("POSTs the workflow definition and returns the imported workflow", async () => {
    __setIwcCacheForTest([WF]);

    let capturedBody: unknown = null;
    let capturedPath: string | null = null;

    const client = mockClient({
      POST: (path, init) => {
        capturedPath = path;
        capturedBody = init?.body;
        return { data: { id: "wf42", name: "RNA-Seq PE" }, response: { status: 200 } };
      },
    });

    const out = await importWorkflowFromIwc({ trsId: WF.trsID }, ctxWith(client));

    expect(capturedPath).toBe("/api/workflows");
    expect((capturedBody as any).workflow).toEqual(DEFINITION);
    expect(out.id).toBe("wf42");
    expect(out.name).toBe("RNA-Seq PE");
  });

  it("project message names the TRS id that was asked for", async () => {
    // Not the id Galaxy gave the copy, which is in data: the other server's sentence
    // quotes the TRS id, so the line reads the same whatever Galaxy called it.
    const imported = { id: "wf99", name: "My Flow" };
    const meta = importWorkflowFromIwcOp.project!(imported, { trsId: WF.trsID });
    expect(meta.message).toBe(`Successfully imported workflow '${WF.trsID}'`);
  });
});
