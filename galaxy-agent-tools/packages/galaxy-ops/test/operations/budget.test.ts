import { describe, it, expect, beforeEach } from "vitest";
import { getIwcWorkflowsOp } from "../../src/operations/get-iwc-workflows";
import { searchIwcWorkflowsOp } from "../../src/operations/search-iwc-workflows";
import { recommendIwcWorkflowsOp, recommendIwcWorkflows } from "../../src/operations/recommend-iwc-workflows";
import { listHistoryIdsOp } from "../../src/operations/list-history-ids";
import { listUserToolsOp } from "../../src/operations/list-user-tools";
import { listWorkflowsOp } from "../../src/operations/list-workflows";
import { searchToolsByNameOp } from "../../src/operations/search-tools-by-name";
import { searchToolsByKeywordsOp } from "../../src/operations/search-tools-by-keywords";
import { getToolPanelOp } from "../../src/operations/get-tool-panel";
import { runWithEnvelope } from "../../src/operations/registry";
import { __setIwcCacheForTest, __resetIwcCacheForTest } from "../../src/iwc-manifest";
import { mockClient } from "../util/mock-client";
import { walkEveryPage, wireBytes } from "../util/budget";
import { OUTPUT_BUDGET_BYTES } from "../../src/operations/pagination";
import {
  hugeIwcManifest,
  hugeIwcWorkflow,
} from "../util/iwc-fixture";
import {
  hugeToolIndex,
  hugeToolPanel,
  hugeUserToolIndex,
  hugeWorkflowIndex,
} from "../util/tool-fixture";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: unknown): GalaxyContext => ({ client, poll: DEFAULT_POLL } as GalaxyContext);
const serving = (rows: unknown[]) => mockClient({ GET: () => ({ data: rows, response: { status: 200 } }) });
const items = (data: unknown) => (data as { items: unknown[] }).items;
const byId = (row: unknown) => (row as { id: string }).id;

/**
 * Ask each budgeted tool for its ceiling against items no ceiling could have
 * allowed for.
 *
 * The ceilings say what a caller may request; this says what comes back is
 * something a model can read. The Python suite checks the same nine tools the
 * same way (`TestEveryPageFitsTheBudget`), which is the point: a page one surface
 * considers safe is safe on the other. Every walk asserts its own fixture really
 * is over budget first, because a page that fits by accident proves nothing.
 */
describe("every page fits the output budget", () => {
  beforeEach(() => __resetIwcCacheForTest());

  it("search_tools_by_name", async () => {
    const corpus = hugeToolIndex(300);
    const seen = await walkEveryPage({
      op: searchToolsByNameOp as never,
      ctx: ctxWith(serving(corpus)),
      input: (limit, offset) => ({ query: "ゲノム", limit, offset }),
      rows: items,
      id: byId,
      limit: 100,
      label: "search_tools_by_name",
    });
    expect(seen).toHaveLength(corpus.length);
  });

  it("search_tools_by_keywords", async () => {
    const panel = hugeToolPanel(1, 300);
    const seen = await walkEveryPage({
      op: searchToolsByKeywordsOp as never,
      ctx: ctxWith(serving(panel)),
      input: (limit, offset) => ({ keywords: ["ゲノム"], limit, offset }),
      rows: items,
      id: byId,
      limit: 200,
      label: "search_tools_by_keywords",
    });
    expect(seen).toHaveLength(300);
  });

  it("get_tool_panel, listing the sections", async () => {
    const panel = hugeToolPanel(600, 2);
    const seen = await walkEveryPage({
      op: getToolPanelOp as never,
      ctx: ctxWith(serving(panel)),
      input: (limit, offset) => ({ limit, offset }),
      rows: (data) => (data as { entries: unknown[] }).entries,
      id: byId,
      limit: 500,
      label: "get_tool_panel sections",
    });
    expect(seen).toHaveLength(600);
  });

  it("get_tool_panel, listing one section's tools", async () => {
    const panel = hugeToolPanel(1, 300);
    panel[0]!.id = "sec";
    const seen = await walkEveryPage({
      op: getToolPanelOp as never,
      ctx: ctxWith(serving(panel)),
      input: (limit, offset) => ({ sectionId: "sec", limit, offset }),
      rows: (data) => (data as { tools: unknown[] }).tools,
      id: byId,
      limit: 500,
      label: "get_tool_panel section",
    });
    expect(seen).toHaveLength(300);
  });

  it("list_workflows", async () => {
    const corpus = hugeWorkflowIndex(400);
    const seen = await walkEveryPage({
      op: listWorkflowsOp as never,
      ctx: ctxWith(serving(corpus)),
      input: (limit, offset) => ({ limit, offset }),
      rows: items,
      id: byId,
      limit: 200,
      label: "list_workflows",
    });
    expect(seen).toHaveLength(corpus.length);
  });

  it("list_user_tools", async () => {
    const corpus = hugeUserToolIndex(200);
    const seen = await walkEveryPage({
      op: listUserToolsOp as never,
      ctx: ctxWith(serving(corpus)),
      input: (limit, offset) => ({ limit, offset }),
      rows: items,
      id: byId,
      limit: 100,
      label: "list_user_tools",
    });
    expect(seen).toHaveLength(corpus.length);
  });

  it("list_history_ids", async () => {
    // Two fields a row, so the rows are small -- but a name is a name, and an
    // account whose histories are called what an imported study calls them puts
    // a page of 500 over the budget that a ceiling of 500 does not bound.
    const corpus = Array.from({ length: 1000 }, (_, k) => ({
      id: String(k).padStart(16, "a"),
      name: `ゲノム解析パイプライン`.repeat(4) + ` (imported from GSE123456, replicate ${k})`,
    }));
    const seen = await walkEveryPage({
      op: listHistoryIdsOp as never,
      ctx: ctxWith(serving(corpus)),
      input: (limit, offset) => ({ limit, offset }),
      rows: items,
      id: byId,
      limit: 500,
      label: "list_history_ids",
    });
    expect(seen).toHaveLength(corpus.length);
  });

  it("get_iwc_workflows", async () => {
    __setIwcCacheForTest(hugeIwcManifest(200));
    const seen = await walkEveryPage({
      op: getIwcWorkflowsOp as never,
      ctx: ctxWith(mockClient({})),
      input: (limit, offset) => ({ limit, offset }),
      rows: items,
      id: (row) => (row as { trsID: string }).trsID,
      limit: 100,
      label: "get_iwc_workflows",
    });
    expect(seen).toHaveLength(200);
  });

  it("search_iwc_workflows", async () => {
    __setIwcCacheForTest(hugeIwcManifest(200));
    const seen = await walkEveryPage({
      op: searchIwcWorkflowsOp as never,
      ctx: ctxWith(mockClient({})),
      input: (limit, offset) => ({ query: "rnaseq", limit, offset }),
      rows: items,
      id: (row) => (row as { trsID: string }).trsID,
      limit: 100,
      label: "search_iwc_workflows",
    });
    expect(seen).toHaveLength(200);
  });

  /**
   * Recommendations have no offset, so there is no walk: a ranking that will not
   * fit is cut from the bottom and says so, which is what the Python tool does.
   */
  it("recommend_iwc_workflows cuts the ranking rather than overrunning", async () => {
    // A quarter of the corpus on topic. BM25's inverse document frequency is zero
    // for a term half the documents share and negative above that, so a query
    // every document answers ranks nothing at all -- the minority is what makes
    // this a ranking rather than an empty list.
    const onTopic = 25;
    __setIwcCacheForTest(
      Array.from({ length: 100 }, (_, k) => {
        const wf = hugeIwcWorkflow(k + 1);
        const tags = k % 4 === 0 ? ["chipseq", "macs2"] : ["proteomics", "maxquant"];
        return { ...wf, definition: { ...wf.definition, tags } };
      }),
    );
    const input = { intent: "chipseq", limit: 25 };
    const ctx = ctxWith(mockClient({}));

    const uncut = await recommendIwcWorkflows(input, ctx);
    expect(uncut.items).toHaveLength(onTopic);

    const result = await runWithEnvelope(recommendIwcWorkflowsOp as never, input as never, ctx);
    const page = result.data as { items: unknown[]; pagination: { trimmedForSize?: boolean; helperText: string } };
    expect(wireBytes(result)).toBeLessThanOrEqual(OUTPUT_BUDGET_BYTES);
    expect(page.items.length).toBeLessThan(onTopic);
    expect(page.pagination.trimmedForSize).toBe(true);
    expect(page.pagination.helperText).toContain("output budget");
  });
});
