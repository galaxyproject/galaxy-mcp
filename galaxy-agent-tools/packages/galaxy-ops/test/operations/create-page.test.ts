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

  it("rejects a standalone report with no slug before calling Galaxy, and says which field", async () => {
    const POST = vi.fn();
    await expect(createPage({ title: "My Report" }, ctxWith(mockClient({ POST })))).rejects.toThrow(
      /a unique slug/,
    );
    expect(POST).not.toHaveBeenCalled();
  });

  it("rejects a standalone report with no title before calling Galaxy", async () => {
    const POST = vi.fn();
    await expect(createPage({ slug: "my-report" }, ctxWith(mockClient({ POST })))).rejects.toBeInstanceOf(
      GalaxyConnectionError,
    );
    expect(POST).not.toHaveBeenCalled();
  });

  it("names both fields when a report supplies neither", async () => {
    const POST = vi.fn();
    await expect(createPage({}, ctxWith(mockClient({ POST })))).rejects.toThrow(
      "a standalone report needs a title and a unique slug; pass a historyId to create a notebook instead",
    );
    expect(POST).not.toHaveBeenCalled();
  });

  it("treats empty strings as absent rather than posting them", async () => {
    const POST = vi.fn();
    await expect(
      createPage({ historyId: "", title: "", slug: "" }, ctxWith(mockClient({ POST }))),
    ).rejects.toBeInstanceOf(GalaxyConnectionError);
    expect(POST).not.toHaveBeenCalled();
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
