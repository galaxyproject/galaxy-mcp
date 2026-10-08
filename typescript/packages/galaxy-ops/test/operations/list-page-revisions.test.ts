import { describe, it, expect } from "vitest";
import { listPageRevisionsOp, listPageRevisions } from "../../src/operations/list-page-revisions";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });
const T0 = "2026-01-01T00:00:00";
const T1 = "2026-01-02T00:00:00";

describe("list_page_revisions", () => {
  it("lists oldest-first by default and carries each revision's edit_source", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/pages/{id}/revisions");
        expect(init.params.path.id).toBe("page1");
        expect(init.params.query.sort_desc).toBe(false);
        return {
          data: [
            { id: "rev1", page_id: "page1", edit_source: "user", create_time: T0, update_time: T0 },
            { id: "rev2", page_id: "page1", edit_source: "agent", create_time: T1, update_time: T1 },
          ],
          response: { status: 200 },
        };
      },
    });
    const out = await listPageRevisions({ pageId: "page1" }, ctxWith(client));
    expect(out).toHaveLength(2);
    expect(out[1]?.edit_source).toBe("agent");
    expect(out[0]?.create_time).toBe(T0);
  });

  it("asks for newest-first when sortDesc is set", async () => {
    const client = mockClient({
      GET: (_path, init) => {
        expect(init.params.query.sort_desc).toBe(true);
        return { data: [], response: { status: 200 } };
      },
    });
    await listPageRevisions({ pageId: "page1", sortDesc: true }, ctxWith(client));
  });

  it("envelopes a missing page as not_found", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "gone" }, response: { status: 404 } }) });
    const r = await runWithEnvelope(listPageRevisionsOp as any, { pageId: "nope" }, ctxWith(client));
    expect(r.success).toBe(false);
    expect(r.errorKind).toBe("not_found");
  });
});
