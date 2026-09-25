import { describe, it, expect } from "vitest";
import { listWorkflowsOp, listWorkflows } from "../../src/operations/list-workflows";
import { mockClient } from "../util/mock-client";
import { workflowIndex } from "../util/tool-fixture";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("list_workflows", () => {
  it("lists workflows and filters the exact name client-side", async () => {
    const client = mockClient({
      GET: (path) => {
        expect(path).toBe("/api/workflows");
        return { data: [{ id: "w1", name: "RNAseq" }, { id: "w2", name: "VarCall" }], response: { status: 200 } };
      },
    });
    const { items: out } = await listWorkflows({ name: "RNAseq" }, ctxWith(client));
    expect((out as any[]).map((w) => w.id)).toEqual(["w1"]);
  });

  it("matches the whole name, because bioblend compares for equality", async () => {
    const client = mockClient({
      GET: () => ({ data: [{ id: "w1", name: "RNAseq" }], response: { status: 200 } }),
    });
    expect((await listWorkflows({ name: "rna" }, ctxWith(client))).items).toEqual([]);
  });
});

describe("list_workflows paging", () => {
  const serving = (n: number) => mockClient({ GET: () => ({ data: workflowIndex(n), response: { status: 200 } }) });

  it("returns the default page of 50 and points at the next", async () => {
    const out = await listWorkflows({}, ctxWith(serving(200)));
    expect(out.items).toHaveLength(50);
    expect(out.pagination).toMatchObject({ total: 200, returned: 50, hasNext: true, nextOffset: 50 });
  });

  it("honours an explicit page", async () => {
    const out = await listWorkflows({ limit: 10, offset: 195 }, ctxWith(serving(200)));
    expect(out.items).toHaveLength(5);
    expect(out.pagination).toMatchObject({ hasNext: false, hasPrevious: true });
  });

  it("pages the name-filtered set, not the unfiltered one", async () => {
    const client = mockClient({
      GET: () => ({
        data: [...workflowIndex(60), { id: "odd1", name: "ChIP-seq peak calling" }],
        response: { status: 200 },
      }),
    });
    const out = await listWorkflows({ name: "ChIP-seq peak calling" }, ctxWith(client));
    expect(out.items).toHaveLength(1);
    expect(out.pagination).toMatchObject({ total: 1, hasNext: false });
  });

  it("rejects a limit over the ceiling the Python tool sets, without calling Galaxy", async () => {
    const client = mockClient({ GET: () => { throw new Error("should not reach Galaxy"); } });
    await expect(listWorkflows({ limit: 201 }, ctxWith(client))).rejects.toThrow(/at most 200/);
  });

  it("handles a user with no workflows", async () => {
    const out = await listWorkflows({}, ctxWith(serving(0)));
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, hasNext: false });
  });

});
