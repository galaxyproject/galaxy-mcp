import { z } from "zod";
import type { GalaxyContext } from "../context";
import { legacyDelete } from "../legacy";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

export interface DeletedUserTool {
  uuid: string;
  deactivated: true;
}

const input = {
  uuid: z.string().describe("the user tool uuid"),
};
type In = { uuid: string };

async function run(i: In, ctx: GalaxyContext): Promise<DeletedUserTool> {
  await legacyDelete<unknown>(ctx, "/api/unprivileged_tools/{uuid}", {
    params: { path: { uuid: i.uuid } },
  });
  return { uuid: i.uuid, deactivated: true };
}

export const deleteUserToolOp: Operation<typeof input, DeletedUserTool> = {
  name: "delete_user_tool",
  domain: "userTools",
  summary: "Deactivate a user-defined tool by uuid (soft delete).",
  input,
  readOnly: false,
  destructive: true,
  run,
  project: (_out, i) => ({ message: `Deactivated user tool ${i.uuid}` }),
};

register(deleteUserToolOp as AnyOperation);

export const deleteUserTool = (i: In, ctx: GalaxyContext) => runOperation(deleteUserToolOp, i, ctx);
