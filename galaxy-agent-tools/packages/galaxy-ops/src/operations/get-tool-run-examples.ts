import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyGet } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export interface ToolRunExamples {
  tool_id: string;
  requested_version: string | null;
  test_cases: unknown[];
}

const input = {
  toolId: z.string().describe("tool id"),
  toolVersion: z.string().optional().describe("specific tool version to fetch test cases for"),
};
type In = { toolId: string; toolVersion?: string };

async function run(i: In, ctx: GalaxyContext): Promise<ToolRunExamples> {
  const testCases = await legacyGet<unknown[]>(ctx, "/api/tools/{tool_id}/test_data", {
    params: {
      path: { tool_id: i.toolId },
      query: i.toolVersion != null ? { tool_version: i.toolVersion } : {},
    },
  });
  return {
    tool_id: i.toolId,
    // Always stated: an absent key reads as "no such field", null as "no version pinned".
    requested_version: i.toolVersion ?? null,
    test_cases: testCases,
  };
}

export const getToolRunExamplesOp: Operation<typeof input, ToolRunExamples> = {
  name: "get_tool_run_examples",
  domain: "tools",
  summary: "Return test-data examples (inputs/outputs) for a Galaxy tool.",
  input,
  run,
  project: (out, i) => ({
    // server.py, get_tool_run_examples.
    message: `Retrieved ${out.test_cases.length} test cases for tool '${i.toolId}'`,
    count: out.test_cases.length,
  }),
  // server.py, get_tool_run_examples: the version joins the context only when one was asked
  // for, which is what `if tool_version:` does there.
  failure: {
    shape: "bioblend-get",
    action: "Get tool run examples",
    context: (i) => ({
      tool_id: i.toolId,
      ...(i.toolVersion ? { tool_version: i.toolVersion } : {}),
    }),
  },
};

register(getToolRunExamplesOp as AnyOperation);

export const getToolRunExamples = (i: In, ctx: GalaxyContext) => runOperation(getToolRunExamplesOp, i, ctx);
