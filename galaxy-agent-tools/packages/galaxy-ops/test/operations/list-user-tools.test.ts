import { describe, it, expect } from "vitest";
import { listUserToolsOp, listUserTools } from "../../src/operations/list-user-tools";
import { mockClient } from "../util/mock-client";
import { paginate } from "../../src/operations/pagination";
import { userToolIndex } from "../util/tool-fixture";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const TOOLS = [
  { id: "1", uuid: "aaaa", tool_id: "my_tool/0.1.0", active: true },
  { id: "2", uuid: "bbbb", tool_id: "other_tool/0.1.0", active: true },
];

describe("list_user_tools", () => {
  it("GETs /api/unprivileged_tools with active:true by default", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/unprivileged_tools");
        expect(init.params.query.active).toBe(true);
        return { data: TOOLS, response: { status: 200 } };
      },
    });
    const { items: out } = await listUserTools({}, ctxWith(client));
    expect(out).toHaveLength(2);
    expect(out[0].uuid).toBe("aaaa");
  });

  it("passes active:false when explicitly set", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(init.params.query.active).toBe(false);
        return { data: [], response: { status: 200 } };
      },
    });
    const { items: out } = await listUserTools({ active: false }, ctxWith(client));
    expect(out).toHaveLength(0);
  });

  it("projects the count message", () => {
    const paged = paginate(TOOLS, { limit: 25, offset: 0, noun: "tools" });
    expect(listUserToolsOp.project!(paged, {} as never).message).toBe("2 of 2 user-defined tool(s)");
  });

  it("throws on HTTP error", async () => {
    const client = mockClient({
      GET: () => ({ error: "server error", response: { status: 500 } }),
    });
    await expect(listUserTools({}, ctxWith(client))).rejects.toThrow();
  });
});

describe("list_user_tools paging", () => {
  const serving = (n: number) => mockClient({ GET: () => ({ data: userToolIndex(n), response: { status: 200 } }) });

  it("returns the default page of 25 and points at the next", async () => {
    const out = await listUserTools({}, ctxWith(serving(100)));
    expect(out.items).toHaveLength(25);
    expect(out.pagination).toMatchObject({ total: 100, returned: 25, hasNext: true, nextOffset: 25 });
  });

  it("honours an explicit page", async () => {
    const out = await listUserTools({ limit: 10, offset: 95 }, ctxWith(serving(100)));
    expect(out.items).toHaveLength(5);
    expect(out.pagination).toMatchObject({ hasNext: false, hasPrevious: true });
  });

  it("rejects a limit over the ceiling the Python tool sets, without calling Galaxy", async () => {
    const client = mockClient({ GET: () => { throw new Error("should not reach Galaxy"); } });
    await expect(listUserTools({ limit: 101 }, ctxWith(client))).rejects.toThrow(/at most 100/);
  });

  it("handles a user with no tools of their own", async () => {
    const out = await listUserTools({}, ctxWith(serving(0)));
    expect(out.items).toEqual([]);
    expect(out.pagination).toMatchObject({ total: 0, hasNext: false });
  });

});
