import { describe, it, expect } from "vitest";
import { revertPageRevisionOp, revertPageRevision } from "../../src/operations/revert-page-revision";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("revert_page_revision", () => {
  it("is a write op but not destructive -- the revision history is append-only", () => {
    expect(revertPageRevisionOp.readOnly).toBe(false);
    expect(revertPageRevisionOp.destructive ?? false).toBe(false);
  });

  it("posts to the revert endpoint and returns the new restored revision", async () => {
    const client = mockClient({
      POST: (path, init) => {
        expect(path).toBe("/api/pages/{id}/revisions/{revision_id}/revert");
        expect(init.params.path.id).toBe("page1");
        expect(init.params.path.revision_id).toBe("rev1");
        return {
          data: {
            id: "rev3",
            page_id: "page1",
            content_editor: "history_dataset_name(history_dataset_id=f2db41e1fa331b3e)",
            content: "Input data",
            edit_source: "restore",
            create_time: "2026-01-03T00:00:00",
            update_time: "2026-01-03T00:00:00",
          },
          response: { status: 200 },
        };
      },
    });
    const out = await revertPageRevision({ pageId: "page1", revisionId: "rev1" }, ctxWith(client));
    expect(out.id).toBe("rev3");
    expect(out.edit_source).toBe("restore");
    expect(out.content_editor).toBe("history_dataset_name(history_dataset_id=f2db41e1fa331b3e)");
    expect(out.content).toBe("Input data");
  });

  it("falls back to content on a server that sends no content_editor", async () => {
    const client = mockClient({
      POST: () => ({
        data: {
          id: "rev3",
          page_id: "page1",
          content: "Input data",
          edit_source: "restore",
          create_time: "2026-01-03T00:00:00",
          update_time: "2026-01-03T00:00:00",
        },
        response: { status: 200 },
      }),
    });
    const out = await revertPageRevision({ pageId: "page1", revisionId: "rev1" }, ctxWith(client));
    expect(out.content_editor).toBe("Input data");
    expect(out.content).toBe("Input data");
  });

  it("envelopes a missing revision as not_found", async () => {
    const client = mockClient({ POST: () => ({ error: { err_msg: "gone" }, response: { status: 404 } }) });
    const r = await runWithEnvelope(
      revertPageRevisionOp as any,
      { pageId: "page1", revisionId: "nope" },
      ctxWith(client),
    );
    expect(r.success).toBe(false);
    expect(r.errorKind).toBe("not_found");
  });
});
