import { describe, it, expect, vi } from "vitest";
import { updatePageOp, updatePage } from "../../src/operations/update-page";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyConnectionError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("update_page", () => {
  it("is a write op", () => {
    expect(updatePageOp.readOnly).toBe(false);
  });

  it("attributes the new revision to the agent and drops the rendered form", async () => {
    const client = mockClient({
      PUT: (path, init) => {
        expect(path).toBe("/api/pages/{id}");
        expect(init.params.path.id).toBe("page1");
        expect(init.body.edit_source).toBe("agent");
        expect(init.body.content).toBe("# updated");
        expect(init.body.title).toBeUndefined();
        return {
          data: { id: "page1", title: "Notebook 1", content: "<rendered>", content_editor: "# updated" },
          response: { status: 200 },
        };
      },
    });
    const out = await updatePage({ pageId: "page1", content: "# updated" }, ctxWith(client));
    expect(out.content_editor).toBe("# updated");
    expect("content" in out).toBe(false);
  });

  it("sends a title-only edit", async () => {
    const client = mockClient({
      PUT: (_path, init) => {
        expect(init.body.title).toBe("Renamed");
        expect(init.body.content).toBeUndefined();
        return { data: { id: "page1", title: "Renamed" }, response: { status: 200 } };
      },
    });
    await updatePage({ pageId: "page1", title: "Renamed" }, ctxWith(client));
  });

  it("refuses an edit that changes nothing", async () => {
    const PUT = vi.fn();
    await expect(updatePage({ pageId: "page1" }, ctxWith(mockClient({ PUT })))).rejects.toBeInstanceOf(
      GalaxyConnectionError,
    );
    expect(PUT).not.toHaveBeenCalled();
  });

  it("envelopes a missing page as not_found", async () => {
    const client = mockClient({ PUT: () => ({ error: { err_msg: "gone" }, response: { status: 404 } }) });
    const r = await runWithEnvelope(updatePageOp as any, { pageId: "nope", content: "x" }, ctxWith(client));
    expect(r.success).toBe(false);
    expect(r.errorKind).toBe("not_found");
  });
});
