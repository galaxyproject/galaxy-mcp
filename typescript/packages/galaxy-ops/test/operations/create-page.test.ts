import { describe, it, expect, vi } from "vitest";
import { createPageOp, createPage } from "../../src/operations/create-page";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import { GalaxyConnectionError } from "../../src/errors";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

describe("create_page", () => {
  it("is a write op", () => {
    expect(createPageOp.readOnly).toBe(false);
  });

  it("creates a notebook attached to a history, in markdown", async () => {
    const client = mockClient({
      POST: (path, init) => {
        expect(path).toBe("/api/pages");
        expect(init.body.content_format).toBe("markdown");
        expect(init.body.history_id).toBe("hist1");
        expect(init.body.content).toBe("# notes");
        // A notebook needs neither -- Galaxy names it -- so we must not invent them.
        expect(init.body.title).toBeUndefined();
        expect(init.body.slug).toBeUndefined();
        return {
          data: { id: "page1", title: "My History", content: "<rendered>", content_editor: "# notes" },
          response: { status: 200 },
        };
      },
    });
    const out = await createPage({ historyId: "hist1", content: "# notes" }, ctxWith(client));
    expect(out.id).toBe("page1");
    // create hands back the editable form, never the rendered one
    expect("content" in out).toBe(false);
  });

  it("creates a standalone report from a title and slug", async () => {
    const client = mockClient({
      POST: (_path, init) => {
        expect(init.body.title).toBe("My Report");
        expect(init.body.slug).toBe("my-report");
        expect(init.body.history_id).toBeUndefined();
        return { data: { id: "page9", title: "My Report", content_editor: "" }, response: { status: 200 } };
      },
    });
    const out = await createPage({ title: "My Report", slug: "my-report" }, ctxWith(client));
    expect(out.id).toBe("page9");
  });

  it("passes an annotation through", async () => {
    const client = mockClient({
      POST: (_path, init) => {
        expect(init.body.annotation).toBe("scratch notes");
        return { data: { id: "page2" }, response: { status: 200 } };
      },
    });
    await createPage({ title: "R", slug: "r", annotation: "scratch notes" }, ctxWith(client));
  });

  // The other server refuses none of these locally: it posts what it was given and lets
  // Galaxy answer. A report with no title and no slug is a 400 from Galaxy with Galaxy's own
  // reason in it, which is what these now check.
  it("posts a report with no slug and lets Galaxy refuse it", async () => {
    const POST = vi.fn(() => ({
      error: { err_msg: "slug is required" },
      response: { status: 400 },
    }));
    await expect(
      createPage({ title: "My Report" }, ctxWith(mockClient({ POST }))),
    ).rejects.toBeInstanceOf(GalaxyConnectionError);
    expect(POST).toHaveBeenCalledOnce();
  });

  it("posts a report with no title for the same reason", async () => {
    const POST = vi.fn(() => ({ error: { err_msg: "title is required" }, response: { status: 400 } }));
    await expect(
      createPage({ slug: "my-report" }, ctxWith(mockClient({ POST }))),
    ).rejects.toBeInstanceOf(GalaxyConnectionError);
    expect(POST).toHaveBeenCalledOnce();
  });

  it("still treats an empty history id as absent, so an empty one makes a report", async () => {
    const POST = vi.fn(() => ({ data: { id: "p1", content_format: "markdown" }, response: { status: 200 } }));
    await createPage({ historyId: "", title: "T", slug: "s" }, ctxWith(mockClient({ POST })));
    const body = POST.mock.calls[0]?.[1]?.body as Record<string, unknown>;
    expect(body).not.toHaveProperty("history_id");
  });

  it("envelopes a slug collision rather than throwing", async () => {
    const client = mockClient({
      POST: () => ({ error: { err_msg: "slug already in use" }, response: { status: 400 } }),
    });
    const r = await runWithEnvelope(
      createPageOp as any,
      { title: "My Report", slug: "taken" },
      ctxWith(client),
    );
    expect(r.success).toBe(false);
    expect(r.errorKind).toBe("connection");
    expect(r.message).toContain("slug already in use");
  });
});
