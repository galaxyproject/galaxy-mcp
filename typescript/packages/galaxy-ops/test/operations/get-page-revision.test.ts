import { describe, it, expect } from "vitest";
import { getPageRevisionOp, getPageRevision } from "../../src/operations/get-page-revision";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("get_page_revision", () => {
  it("returns editable and expanded revision content without conflating them", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/pages/{id}/revisions/{revision_id}");
        expect(init.params.path.id).toBe("page1");
        expect(init.params.path.revision_id).toBe("rev1");
        return {
          data: {
            id: "rev1",
            page_id: "page1",
            content_editor: "history_dataset_name(history_dataset_id=f2db41e1fa331b3e)",
            content: "Input data",
            content_format: "markdown",
            edit_source: "user",
            create_time: "2026-01-01T00:00:00",
            update_time: "2026-01-01T00:00:00",
          },
          response: { status: 200 },
        };
      },
    });
    const out = await getPageRevision({ pageId: "page1", revisionId: "rev1" }, ctxWith(client));
    expect(out.content_editor).toBe("history_dataset_name(history_dataset_id=f2db41e1fa331b3e)");
    expect(out.content).toBe("Input data");
    expect(out.edit_source).toBe("user");
    expect(out.content_editor_source).toBe("server");
  });

  it("falls back to content on a server that sends no content_editor", async () => {
    // Galaxy through 26.1.1 leaves content_editor off the revision response model, so the
    // expanded export form is the only body there is.
    const client = mockClient({
      GET: () => ({
        data: {
          id: "rev1",
          page_id: "page1",
          content: "Input data",
          content_format: "markdown",
          edit_source: "user",
          create_time: "2026-01-01T00:00:00",
          update_time: "2026-01-01T00:00:00",
        },
        response: { status: 200 },
      }),
    });
    const out = await getPageRevision({ pageId: "page1", revisionId: "rev1" }, ctxWith(client));
    expect(out.content_editor).toBe("Input data");
    expect(out.content).toBe("Input data");
    // The caller is told, rather than left to notice the two bodies are identical.
    expect(out.content_editor_source).toBe("content");
  });

  it("treats a null content_editor the same as a missing one", async () => {
    const client = mockClient({
      GET: () => ({
        data: {
          id: "rev1",
          page_id: "page1",
          content_editor: null,
          content: "Input data",
          create_time: "2026-01-01T00:00:00",
          update_time: "2026-01-01T00:00:00",
        },
        response: { status: 200 },
      }),
    });
    const out = await getPageRevision({ pageId: "page1", revisionId: "rev1" }, ctxWith(client));
    expect(out.content_editor).toBe("Input data");
    expect(out.content_editor_source).toBe("content");
  });

  it("treats an empty content_editor as missing, which is what an html revision sends", async () => {
    const client = mockClient({
      GET: () => ({
        data: {
          id: "rev1",
          page_id: "page1",
          content_editor: "",
          content: "<h1>Results</h1>",
          content_format: "html",
          create_time: "2026-01-01T00:00:00",
          update_time: "2026-01-01T00:00:00",
        },
        response: { status: 200 },
      }),
    });
    const out = await getPageRevision({ pageId: "page1", revisionId: "rev1" }, ctxWith(client));
    expect(out.content_editor).toBe("<h1>Results</h1>");
    expect(out.content_editor_source).toBe("content");
  });

  it("counts an empty content as a body, because that is what the caller gets", async () => {
    const client = mockClient({
      GET: () => ({
        data: {
          id: "rev1",
          page_id: "page1",
          content_editor: "",
          content: "",
          create_time: "2026-01-01T00:00:00",
          update_time: "2026-01-01T00:00:00",
        },
        response: { status: 200 },
      }),
    });
    const out = await getPageRevision({ pageId: "page1", revisionId: "rev1" }, ctxWith(client));
    expect(out.content_editor).toBe("");
    expect(out.content_editor_source).toBe("content");
  });

  it("says so when the revision carried no body at all", async () => {
    const client = mockClient({
      GET: () => ({
        data: {
          id: "rev1",
          page_id: "page1",
          create_time: "2026-01-01T00:00:00",
          update_time: "2026-01-01T00:00:00",
        },
        response: { status: 200 },
      }),
    });
    const out = await getPageRevision({ pageId: "page1", revisionId: "rev1" }, ctxWith(client));
    expect(out.content_editor).toBeNull();
    expect(out.content_editor_source).toBe("none");
  });

  it("envelopes a missing revision as not_found", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "gone" }, response: { status: 404 } }) });
    const r = await runWithEnvelope(
      getPageRevisionOp as any,
      { pageId: "page1", revisionId: "nope" },
      ctxWith(client),
    );
    expect(r.success).toBe(false);
    expect(r.errorKind).toBe("not_found");
  });
});
