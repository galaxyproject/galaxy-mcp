import { z } from "zod";
import type { GalaxyContext } from "../context";
import { GalaxyConnectionError } from "../errors";
import { legacyPost } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/** Hand-typed: created user-defined tool record from POST /api/unprivileged_tools. */
export interface CreatedUserTool {
  id?: string;
  uuid?: string;
  tool_id?: string;
  active?: boolean;
  [k: string]: unknown;
}

const REQUIRED_FIELDS = ["class", "id", "version", "name", "shell_command", "container"] as const;

const input = {
  representation: z
    .record(z.string(), z.unknown())
    .describe(
      "a GalaxyUserTool representation: {class:'GalaxyUserTool', id, version, name, shell_command, container:'<image>'}",
    ),
};
type In = { representation: Record<string, unknown> };

function validate(rep: Record<string, unknown>): void {
  for (const field of REQUIRED_FIELDS) {
    if (!(field in rep)) {
      throw new GalaxyConnectionError(`representation is missing required field: '${field}'`, 400);
    }
  }
  if (rep["class"] !== "GalaxyUserTool") {
    throw new GalaxyConnectionError(
      `class must be 'GalaxyUserTool', got '${String(rep["class"])}'`,
      400,
    );
  }
  if (typeof rep["container"] !== "string") {
    throw new GalaxyConnectionError(
      `container must be a string (e.g. 'python:3.12-slim'), got ${typeof rep["container"]}: ${JSON.stringify(rep["container"])}`,
      400,
    );
  }
}

async function run(i: In, ctx: GalaxyContext): Promise<CreatedUserTool> {
  validate(i.representation);
  return legacyPost<CreatedUserTool>(ctx, "/api/unprivileged_tools", {
    body: { src: "representation", representation: i.representation },
  });
}

export const createUserToolOp: Operation<typeof input, CreatedUserTool> = {
  name: "create_user_tool",
  domain: "userTools",
  summary: "Create a user-defined tool in Galaxy from a tool representation dict.",
  input,
  readOnly: false,
  run,
  project: (out) => ({ message: `Created user tool ${out.uuid ?? out.id ?? ""}` }),
};

register(createUserToolOp as AnyOperation);

export const createUserTool = (i: In, ctx: GalaxyContext) => runOperation(createUserToolOp, i, ctx);
