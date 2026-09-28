import type { GalaxyContext } from "../context";
import { GalaxyConnectionError } from "../errors";
import { jsonObject } from "../json-object";
import { legacyPost } from "../legacy";
import { pyGet, pyStr } from "../python-values";
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
  representation: jsonObject().describe(
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
  // server.py, create_user_tool: the name out of the REPRESENTATION that was sent,
  // falling back to its id and then to "unknown" -- both reads are dict.get, so a
  // representation stating a null name is announced as None rather than defaulted.
  // The uuid Galaxy assigned is in data, which is what delete_user_tool takes.
  project: (_out, i) => ({
    message: `Created user-defined tool '${pyStr(
      pyGet(i.representation, "name", pyGet(i.representation, "id", "unknown")),
    )}'`,
  }),
  // server.py, create_user_tool: the context names the id out of the representation with
  // dict.get, so a representation without one reads None rather than being defaulted.
  failure: {
    shape: "bioblend-write",
    action: "Create user tool",
    context: (i) => ({ tool_id: pyGet(i.representation as Record<string, unknown>, "id", undefined) }),
  },
};

register(createUserToolOp as AnyOperation);

export const createUserTool = (i: In, ctx: GalaxyContext) => runOperation(createUserToolOp, i, ctx);
