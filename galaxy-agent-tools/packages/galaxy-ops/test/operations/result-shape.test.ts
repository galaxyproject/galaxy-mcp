/** Every declared result shape is the shape the operation actually returns.
 *
 * `result` is what a caller is told it may read. Nothing else checks it: the types say the shape
 * at compile time, and a projection that quietly stops building one of its keys still satisfies
 * an index signature. get_tool_panel shipped for a release without the tool_count its own summary
 * promised, and a scenario found it rather than a test.
 */
import { describe, it, expect, beforeEach } from "vitest";
import "../../src/operations/all";
import { allOperations } from "../../src/operations/registry";
import { __resetIwcCacheForTest, __setIwcCacheForTest } from "../../src/iwc-manifest";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

/** One panel section holding one tool, which every panel projection has to cope with. */
const PANEL = [
  { id: "sec", name: "Section", elems: [{ id: "cat1", name: "Concatenate", description: "join" }] },
  { id: "loose", name: "Loose tool", description: "outside a section" },
];

/** Routes that answer with a collection; everything else answers with the one object below. */
const COLLECTIONS = [
  "/api/histories",
  "/api/histories/{history_id}/contents",
  "/api/workflows",
  "/api/unprivileged_tools",
  "/api/pages",
  "/api/pages/{id}/revisions",
];

/** Answers every route a declaring op reads, so one mock serves the whole table. */
function client(): GalaxyContext {
  const body = {
    id: "x1",
    name: "thing",
    version: "1.0",
    state: "ok",
    file_ext: "txt",
    count: 3,
    hid: 1,
    citations: [],
    inputs: [],
    elements: [],
    element_count: 0,
    populated: true,
    collection_type: "list",
    job: { id: "j1", state: "ok" },
    // get_job_details reads the job id off the dataset before it reads the job.
    creating_job: "j1",
  };
  return {
    client: mockClient({
      GET: (path: string, init?: any) => {
        if (path === "/api/tools") return { data: PANEL, response: { status: 200 } };
        if (path.endsWith("/test_data")) return { data: [], response: { status: 200 } };
        if (init?.parseAs === "arrayBuffer") {
          return { data: new TextEncoder().encode("a\nb").buffer, response: { status: 200 } };
        }
        if (COLLECTIONS.includes(path)) return { data: [body], response: { status: 200 } };
        return { data: body, response: { status: 200 } };
      },
      DELETE: () => ({ data: {}, response: { status: 200 } }),
    }),
    poll: DEFAULT_POLL,
  } as GalaxyContext;
}

/** The smallest arguments each declaring op needs; a new declaration without one fails below. */
const ARGS: Record<string, Record<string, unknown>> = {
  get_tool_panel: {},
  get_collection_details: { collectionId: "c1" },
  get_history_details: { historyId: "h1" },
  get_history_contents: { historyId: "h1" },
  get_histories: {},
  list_history_ids: {},
  get_tool_run_examples: { toolId: "cat1" },
  get_tool_input_template: { toolId: "cat1" },
  get_tool_citations: { toolId: "cat1" },
  get_job_details: { datasetId: "d1" },
  cancel_workflow_invocation: { invocationId: "i1" },
  delete_user_tool: { uuid: "u1" },
  download_dataset: { datasetId: "d1" },
  list_workflows: {},
  list_user_tools: {},
  list_pages: {},
  list_page_revisions: { pageId: "p1" },
  search_tools_by_name: { query: "cat" },
  search_tools_by_keywords: { keywords: ["cat"] },
  get_iwc_workflows: {},
  search_iwc_workflows: { query: "alpha" },
  recommend_iwc_workflows: { historyId: "h1" },
  get_server_info: {},
};

const declaring = allOperations.filter((op) => op.result !== undefined);

// The IWC ops read a remote manifest rather than the Galaxy client; the cache stands in for it.
beforeEach(() => {
  __resetIwcCacheForTest();
  __setIwcCacheForTest([
    { trsID: "#workflow/github.com/iwc-workflows/alpha/main", definition: { name: "Alpha" } },
  ]);
});

describe("declared result shapes", () => {
  it("covers every op that declares one", () => {
    const missing = declaring.map((op) => op.name).filter((name) => !(name in ARGS));
    expect(missing, "declared a result shape with no way to read it back").toEqual([]);
  });

  it.each(declaring.map((op) => [op.name, op] as const))("%s returns what it declares", async (name, op) => {
    const data = await op.run(ARGS[name] as never, client());
    const shape = op.result!;
    if (shape.kind === "list") {
      expect(Array.isArray(data), `${name} declares a list`).toBe(true);
      return;
    }
    expect(Array.isArray(data), `${name} declares an object`).toBe(false);
    const carried = data as Record<string, unknown>;
    const absent = (shape.fields ?? []).filter((field) => !(field in carried));
    expect(absent, `${name} promises fields its result does not carry`).toEqual([]);
  });

  it.each(declaring.map((op) => [op.name, op] as const))("%s pages where it says", async (name, op) => {
    // `paginated` is what the parity report compares against the other surface, so it is held to
    // the envelope this op really builds rather than left to whoever wrote the declaration.
    const data = await op.run(ARGS[name] as never, client());
    const emitted = op.project?.(data as never, ARGS[name] as never)?.pagination !== undefined;
    expect(emitted, `${name} declares paginated=${Boolean(op.result!.paginated)}`).toBe(
      Boolean(op.result!.paginated),
    );
  });
});
