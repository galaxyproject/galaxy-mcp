import type { GalaxyContext } from "../context";
import { classifyHttp, GalaxyAuthError } from "../errors";
import { register, runOperation } from "./registry";
import type { AnyOperation, Operation } from "./types";

/**
 * The user record Galaxy sent, whole.
 *
 * The Python tool answers with what `/api/users/current` returned and picks no fields
 * out of it, so neither does this: a DetailedUserModel carries the disk usage, the
 * quota, the tags in use and the stored preferences, and a caller reading any of them
 * should not have to ask Galaxy a second time. The three named fields are the ones
 * anything reading a user needs and the ones the guard below insists on; everything
 * else rides along under the index signature.
 */
export type CurrentUser = Record<string, unknown> & {
  id: string;
  email: string;
  username: string;
};

const input = {}; // no args

async function run(_in: Record<string, never>, ctx: GalaxyContext): Promise<CurrentUser> {
  const { data, error, response } = await ctx.client.GET("/api/users/{user_id}", {
    params: { path: { user_id: "current" } },
  });
  if (error || !data) throw classifyHttp(response.status, error);
  // The endpoint returns DetailedUserModel | AnonUserModel; the anonymous shape has
  // no id/email/username. Surface that as an auth error rather than silently
  // returning undefined fields.
  const u = data as { id?: string; email?: string; username?: string };
  if (!u.id || !u.email || !u.username) {
    throw new GalaxyAuthError("Anonymous user response -- a valid API key is required");
  }
  return data as CurrentUser;
}

export const getUserOp: Operation<typeof input, CurrentUser> = {
  name: "get_user", // parity: mcp-server-galaxy-py get_user
  domain: "connection",
  summary: "Return the current authenticated Galaxy user, as Galaxy's own user record.",
  input,
  run,
  project: (u) => ({ message: `Authenticated as ${u.username} <${u.email}>` }),
};

register(getUserOp as AnyOperation);

/** Code-mode entry. */
export const getUser = (i: Record<string, never>, ctx: GalaxyContext) => runOperation(getUserOp, i, ctx);
