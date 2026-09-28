import { describe, it, expect } from "vitest";
import { deleteUserToolOp, deleteUserTool } from "../../src/operations/delete-user-tool";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("delete_user_tool", () => {
  it("is a write op and destructive", () => {
    expect(deleteUserToolOp.readOnly).toBe(false);
    expect(deleteUserToolOp.destructive).toBe(true);
  });

  it("DELETEs /api/unprivileged_tools/{uuid} and returns deactivated result", async () => {
    const client = mockClient({
      DELETE: (path, init) => {
        expect(path).toBe("/api/unprivileged_tools/{uuid}");
        expect(init.params.path.uuid).toBe("aaaa-1111");
        return { data: { uuid: "aaaa-1111", deactivated: true }, response: { status: 200 } };
      },
    });
    const out = await deleteUserTool({ uuid: "aaaa-1111" }, ctxWith(client));
    expect(out.uuid).toBe("aaaa-1111");
    expect(out.deactivated).toBe(true);
  });

  it("projects the uuid into the message", () => {
    expect(deleteUserToolOp.project!({ uuid: "aaaa-1111", deactivated: true }, { uuid: "aaaa-1111" })).toEqual({
      message: "Deactivated user-defined tool 'aaaa-1111'",
    });
  });

  it("throws on DELETE error", async () => {
    const client = mockClient({
      DELETE: () => ({ error: "not found", response: { status: 404 } }),
    });
    await expect(deleteUserTool({ uuid: "bad-uuid" }, ctxWith(client))).rejects.toThrow();
  });
});
