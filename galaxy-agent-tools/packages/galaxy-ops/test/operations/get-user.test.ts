import { describe, it, expect } from "vitest";
import { getUserOp, getUser } from "../../src/operations/get-user";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyAuthError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_user", () => {
  it("has parity name and returns id/email/username", async () => {
    expect(getUserOp.name).toBe("get_user");
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/users/{user_id}");
        expect(init.params.path.user_id).toBe("current");
        return { data: { id: "u1", email: "a@b.c", username: "alice" }, response: { status: 200 } };
      },
    });
    const out = await getUser({}, ctxWith(client));
    expect(out).toEqual({ id: "u1", email: "a@b.c", username: "alice" });
  });

  it("throws GalaxyAuthError on 401", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "no" }, response: { status: 401 } }) });
    await expect(getUser({}, ctxWith(client))).rejects.toBeInstanceOf(GalaxyAuthError);
  });

  it("throws GalaxyAuthError on an anonymous (200) response with no id", async () => {
    const client = mockClient({
      GET: () => ({ data: { total_disk_usage: 0, quota_percent: null }, response: { status: 200 } }),
    });
    await expect(getUser({}, ctxWith(client))).rejects.toBeInstanceOf(GalaxyAuthError);
  });
});
