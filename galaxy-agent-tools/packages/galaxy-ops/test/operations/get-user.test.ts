import { describe, it, expect } from "vitest";
import { getUserOp, getUser } from "../../src/operations/get-user";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyAuthError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_user", () => {
  it("has parity name and hands back the record Galaxy sent", async () => {
    expect(getUserOp.name).toBe("get_user");
    // Every field, not the three the guard reads: the Python tool answers with the whole
    // record, and a caller after the quota or the disk usage has it without asking twice.
    const record = {
      id: "u1",
      email: "a@b.c",
      username: "alice",
      total_disk_usage: 1048576,
      nice_total_disk_usage: "1.0 MB",
      quota_percent: 12.5,
      tags_used: ["rnaseq"],
    };
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/users/{user_id}");
        expect(init.params.path.user_id).toBe("current");
        return { data: record, response: { status: 200 } };
      },
    });
    const out = await getUser({}, ctxWith(client));
    expect(out).toEqual(record);
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
