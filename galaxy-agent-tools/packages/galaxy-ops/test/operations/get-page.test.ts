import { describe, it, expect } from "vitest";
import { getPageOp, getPage } from "../../src/operations/get-page";
import { runWithEnvelope } from "../../src/operations/registry";
import { mockClient } from "../util/mock-client";
import { DEFAULT_POLL } from "../../src/context";
import type { GalaxyContext } from "../../src/context";

const ctxWith = (client: any): GalaxyContext => ({ client, poll: DEFAULT_POLL });

const page = {
  id: "page1",
  title: "Notebook 1",
  content: "<rendered html>",
  content_editor: "raw markdown",
};

describe("get_page", () => {
  it("returns the editable markdown and drops the rendered form", async () => {
    const client = mockClient({
      GET: (path, init) => {
        expect(path).toBe("/api/pages/{id}");
        expect(init.params.path.id).toBe("page1");
        return { data: { ...page }, response: { status: 200 } };
      },
    });
    const out = await getPage({ pageId: "page1" }, ctxWith(client));
    expect(out.content_editor).toBe("raw markdown");
    expect("content" in out).toBe(false);
  });

  it("keeps the rendered form when it is asked for", async () => {
    const client = mockClient({ GET: () => ({ data: { ...page }, response: { status: 200 } }) });
    const out = await getPage({ pageId: "page1", includeRendered: true }, ctxWith(client));
    expect(out.content).toBe("<rendered html>");
    expect(out.content_editor).toBe("raw markdown");
  });

  it("drops content on an html page too, although that is the only body it has", async () => {
    // The other server's drop is unconditional and so is this one. An HTML page's body is
    // in `content` because Galaxy fills content_editor on the markdown path only -- and it
    // is the expanded form either way, which a caller must not edit and send back. Asking
    // for it is what includeRendered is for.
    const html = {
      id: "page2",
      title: "Old report",
      content_format: "html",
      content: "<h1>Results</h1>",
      content_editor: "",
    };
    const client = mockClient({ GET: () => ({ data: html, response: { status: 200 } }) });
    const out = await getPage({ pageId: "page2" }, ctxWith(client));
    expect(out.content).toBeUndefined();
    expect("content" in out).toBe(false);

    const asked = mockClient({ GET: () => ({ data: html, response: { status: 200 } }) });
    const withRender = await getPage({ pageId: "page2", includeRendered: true }, ctxWith(asked));
    expect(withRender.content).toBe("<h1>Results</h1>");
  });

  it("envelopes a missing page as not_found", async () => {
    const client = mockClient({ GET: () => ({ error: { err_msg: "gone" }, response: { status: 404 } }) });
    const r = await runWithEnvelope(getPageOp as any, { pageId: "nope" }, ctxWith(client));
    expect(r.success).toBe(false);
    expect(r.errorKind).toBe("not_found");
  });
});
