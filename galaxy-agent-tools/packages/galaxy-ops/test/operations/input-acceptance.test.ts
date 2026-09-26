import { describe, it, expect } from "vitest";
import { getHistoriesOp, getHistories } from "../../src/operations/get-histories";
import { getHistoryContentsOp } from "../../src/operations/get-history-contents";
import { listHistoryIdsOp } from "../../src/operations/list-history-ids";
import { listUserToolsOp } from "../../src/operations/list-user-tools";
import { listWorkflowsOp, listWorkflows } from "../../src/operations/list-workflows";
import { searchToolsByNameOp } from "../../src/operations/search-tools-by-name";
import { searchToolsByKeywordsOp } from "../../src/operations/search-tools-by-keywords";
import { getToolPanelOp, getToolPanel } from "../../src/operations/get-tool-panel";
import { getIwcWorkflowsOp } from "../../src/operations/get-iwc-workflows";
import { searchIwcWorkflowsOp } from "../../src/operations/search-iwc-workflows";
import { recommendIwcWorkflowsOp } from "../../src/operations/recommend-iwc-workflows";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";
import type { ToolPanelOverview } from "../../src/operations/get-tool-panel";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

/**
 * What these tools accept, against what the Python server's manifest says they
 * accept (`mcp-server-galaxy-py/tests/testdata/mcp-surface.json`).
 *
 * The parity check compares the types, but its normalization folds an optional
 * `T | null` together with a plain optional `T` -- deliberately, because that
 * union is how Python spells optional -- so it cannot see the difference between a
 * parameter that takes an explicit null and one that refuses it. Nor does it look
 * at what a schema coerces. Both are things a client trips over, so they are
 * pinned here instead.
 */
const integerInputs: Array<[string, { safeParse(v: unknown): { success: boolean } }]> = [
  ["get_histories.limit", getHistoriesOp.input.limit],
  ["get_histories.offset", getHistoriesOp.input.offset],
  ["get_history_contents.limit", getHistoryContentsOp.input.limit],
  ["get_history_contents.offset", getHistoryContentsOp.input.offset],
  ["list_history_ids.limit", listHistoryIdsOp.input.limit],
  ["list_history_ids.offset", listHistoryIdsOp.input.offset],
  ["list_user_tools.limit", listUserToolsOp.input.limit],
  ["list_workflows.limit", listWorkflowsOp.input.limit],
  ["search_tools_by_name.limit", searchToolsByNameOp.input.limit],
  ["search_tools_by_name.offset", searchToolsByNameOp.input.offset],
  ["search_tools_by_keywords.limit", searchToolsByKeywordsOp.input.limit],
  ["get_tool_panel.limit", getToolPanelOp.input.limit],
  ["get_iwc_workflows.limit", getIwcWorkflowsOp.input.limit],
  ["search_iwc_workflows.limit", searchIwcWorkflowsOp.input.limit],
  ["recommend_iwc_workflows.limit", recommendIwcWorkflowsOp.input.limit],
];

describe("input acceptance matches the Python manifest", () => {
  it("takes an integer for every integer parameter", () => {
    for (const [where, schema] of integerInputs) {
      expect(schema.safeParse(7).success, where).toBe(true);
    }
  });

  it("refuses what an integer parameter is not, rather than coercing it", () => {
    for (const [where, schema] of integerInputs) {
      expect(schema.safeParse("5").success, `${where} took a string`).toBe(false);
      expect(schema.safeParse([1]).success, `${where} took a list`).toBe(false);
      expect(schema.safeParse(1.5).success, `${where} took a fraction`).toBe(false);
    }
  });

  it("accepts null exactly where Python declares a null branch", () => {
    // `int | None`, `str | None`: null is how a client says "use the default".
    expect(getHistoriesOp.input.limit.safeParse(null).success).toBe(true);
    expect(getHistoriesOp.input.name.safeParse(null).success).toBe(true);
    expect(listWorkflowsOp.input.name.safeParse(null).success).toBe(true);
    expect(getToolPanelOp.input.sectionId.safeParse(null).success).toBe(true);
  });

  it("refuses null everywhere Python declares a plain integer or string", () => {
    expect(getHistoriesOp.input.offset.safeParse(null).success).toBe(false);
    expect(getHistoryContentsOp.input.limit.safeParse(null).success).toBe(false);
    expect(getHistoryContentsOp.input.historyId.safeParse(null).success).toBe(false);
    expect(listHistoryIdsOp.input.offset.safeParse(null).success).toBe(false);
    expect(searchToolsByNameOp.input.limit.safeParse(null).success).toBe(false);
    expect(getToolPanelOp.input.limit.safeParse(null).success).toBe(false);
    expect(recommendIwcWorkflowsOp.input.intent.safeParse(null).success).toBe(false);
  });

  it("reads a null section id as the overview, the way Python reads it", async () => {
    const client = mockClient({
      GET: () => ({
        data: [{ id: "s1", name: "Genomics", model_class: "ToolSection", elems: [{ id: "fastqc" }] }],
        response: { status: 200 },
      }),
    });
    const out = (await getToolPanel({ sectionId: null }, ctxWith(client))) as ToolPanelOverview;
    expect(out.entries).toEqual([{ id: "s1", name: "Genomics", type: "section", tool_count: 1 }]);
  });

  it("reads a null name as no filter, the way an absent one is", async () => {
    const rows = [{ id: "h1", name: "alpha" }, { id: "h2", name: "beta" }];
    const histories = mockClient({ GET: () => ({ data: rows, response: { status: 200 } }) });
    expect((await getHistories({ name: null }, ctxWith(histories))).items).toHaveLength(2);
    const workflows = mockClient({ GET: () => ({ data: rows, response: { status: 200 } }) });
    expect((await listWorkflows({ name: null }, ctxWith(workflows))).items).toHaveLength(2);
  });
});
