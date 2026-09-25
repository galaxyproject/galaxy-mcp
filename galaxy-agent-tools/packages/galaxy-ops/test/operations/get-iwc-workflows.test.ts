import { describe, it, expect, beforeEach } from "vitest";
import { getIwcWorkflowsOp, getIwcWorkflows } from "../../src/operations/get-iwc-workflows";
import { __resetIwcCacheForTest, __setIwcCacheForTest, type IwcWorkflow } from "../../src/iwc-manifest";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import { mockClient } from "../util/mock-client";
import { iwcManifest } from "../util/iwc-fixture";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const WF_A: IwcWorkflow = {
  trsID: "#workflow/github.com/iwc-workflows/alpha/main",
  definition: { name: "Alpha" },
};
const WF_B: IwcWorkflow = {
  trsID: "#workflow/github.com/iwc-workflows/beta/main",
  definition: { name: "Beta" },
};

beforeEach(() => __resetIwcCacheForTest());

describe("get_iwc_workflows", () => {
  it("returns all workflows from the primed cache", async () => {
    __setIwcCacheForTest([WF_A, WF_B]);
    const out = await getIwcWorkflows({}, ctxWith(mockClient({})));
    expect(out.items).toHaveLength(2);
    expect(out.items[0].trsID).toBe(WF_A.trsID);
  });

  it("is read-only by default", () => {
    expect(getIwcWorkflowsOp.readOnly).not.toBe(false);
  });

  it("project returns the count message", async () => {
    __setIwcCacheForTest([WF_A, WF_B]);
    const out = await getIwcWorkflows({}, ctxWith(mockClient({})));
    const meta = getIwcWorkflowsOp.project!(out, {} as never);
    expect(meta.message).toBe("2 of 2 IWC workflows");
  });
});

describe("get_iwc_workflows paging", () => {
  it("returns the default page of 20 summaries and points at the next", async () => {
    __setIwcCacheForTest(iwcManifest(123));
    const out = await getIwcWorkflows({}, ctxWith(mockClient({})));
    expect(out.items).toHaveLength(20);
    expect(out.pagination).toMatchObject({ total: 123, returned: 20, limit: 20, hasNext: true, nextOffset: 20 });
  });

  it("returns summaries, not raw manifest entries", async () => {
    __setIwcCacheForTest(iwcManifest(1));
    const { items } = await getIwcWorkflows({}, ctxWith(mockClient({})));
    expect(items[0]).toHaveProperty("readme_summary");
    expect(items[0]).toHaveProperty("step_count", 24);
    expect(items[0]).not.toHaveProperty("definition");
    expect(items[0]).not.toHaveProperty("readme");
  });

  it("honours an explicit page", async () => {
    __setIwcCacheForTest(iwcManifest(123));
    const out = await getIwcWorkflows({ limit: 5, offset: 120 }, ctxWith(mockClient({})));
    expect(out.items).toHaveLength(3);
    expect(out.pagination).toMatchObject({ hasNext: false, hasPrevious: true });
  });

  it("rejects a limit over the ceiling the Python tool sets, instead of clamping it", async () => {
    __setIwcCacheForTest(iwcManifest(123));
    await expect(getIwcWorkflows({ limit: 101 }, ctxWith(mockClient({})))).rejects.toThrow(/at most 100/);
  });

  it("handles an empty manifest", async () => {
    __setIwcCacheForTest([]);
    const out = await getIwcWorkflows({}, ctxWith(mockClient({})));
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, hasNext: false });
  });

});
