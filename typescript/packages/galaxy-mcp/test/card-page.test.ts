import { describe, expect, it } from "vitest";
import { cardPage, type CardSpec } from "../src/apps/card-page";
import { runCard as runPage, toolResult } from "./card-host";

/** A card over a made-up paged read, so the page's own behaviour is what is under test. */
const spec: CardSpec = {
  uri: "ui://test/things.html",
  title: "Things",
  pageTool: "get_things",
  openTool: "get_things",
  keep: ["filter"],
  read: `(page, kept) => ({
    items: page.data || [],
    total: typeof page.pagination?.total_items === "number" ? page.pagination.total_items : null,
    nextOffset: page.pagination?.has_next ? page.pagination.next_offset : null,
    scope: kept.filter ? "Filtered by " + kept.filter : "All things",
  })`,
  row: `(item) => ({ name: item.name, meta: [item.kind], state: item.state, detail: item.kind === "file" ? [["Kind", item.kind], ["Size", bytes(item.size)]] : undefined })`,
  empty: "Nothing here.",
};

const runCard = () => runPage(cardPage(spec));

const thing = (n: number) => ({ name: `thing ${n}`, kind: n % 2 ? "file" : "folder", state: "ok", size: 2048 });
const things = (from: number, count: number) => Array.from({ length: count }, (_, i) => thing(from + i));
const result = toolResult;

describe("the card page", () => {
  it("initializes first, and reports its size only once the host has it", async () => {
    const card = runCard();
    expect(card.sent).toEqual([
      expect.objectContaining({
        method: "ui/initialize",
        params: expect.objectContaining({ appCapabilities: { availableDisplayModes: ["inline", "fullscreen"] } }),
      }),
    ]);
    await card.initialize();
    expect(card.sent.slice(1).map((m) => m.method)).toEqual(["ui/notifications/initialized", "ui/notifications/size-changed"]);
  });

  it("shows five rows, the scope and Galaxy's total in the header", async () => {
    const card = runCard();
    await card.initialize();
    await card.open({ filter: "big" }, things(1, 20), { total_items: 42, has_next: true, next_offset: 20 });
    expect(card.rows()).toEqual(["thing 1", "thing 2", "thing 3", "thing 4", "thing 5"]);
    expect(card.scope()).toBe("Filtered by big · Showing 5 of 42");
  });

  it("counts what it holds when Galaxy gives no total", async () => {
    const card = runCard();
    await card.initialize();
    await card.open({}, things(1, 7), null);
    expect(card.scope()).toBe("All things · Showing 5 of 7");

    const paged = runCard();
    await paged.initialize();
    await paged.open({}, things(1, 20), { has_next: true, next_offset: 20 });
    expect(paged.scope()).toBe("All things · Showing 5 of 20+");
  });

  it("shows rows it already holds before fetching, then fetches from Galaxy's next_offset", async () => {
    const card = runCard();
    await card.initialize();
    await card.open({ filter: "big", limit: 8 }, things(1, 8), { total_items: 42, has_next: true, next_offset: 8 });
    const before = card.sent.length;
    await card.click("Show more");
    expect(card.rows()).toHaveLength(8);
    expect(card.sent).toHaveLength(before); // nothing fetched yet

    const showing = card.click("Show more");
    const call = card.sent.at(-1)!;
    expect(call).toMatchObject({ method: "tools/call", params: { name: "get_things", arguments: { filter: "big", limit: 20, offset: 8 } } });
    expect(call.params.arguments).not.toHaveProperty("limit", 8);
    await card.deliver({ id: call.id, result: result(things(9, 20), { total_items: 42, has_next: true, next_offset: 28 }) });
    await showing;
    expect(card.rows()).toHaveLength(13);
    expect(card.scope()).toBe("Filtered by big · Showing 13 of 42");
  });

  it("shows less without forgetting rows or moving the cursor", async () => {
    const card = runCard();
    await card.initialize();
    await card.open({}, things(1, 20), { total_items: 42, has_next: true, next_offset: 20 });
    await card.click("Show more");
    await card.click("Show more");
    expect(card.rows()).toHaveLength(15);
    await card.click("Show less");
    expect(card.rows()).toHaveLength(10);
    const before = card.sent.length;
    await card.click("Show more");
    expect(card.rows()).toHaveLength(15);
    expect(card.sent).toHaveLength(before);
    await card.click("Show less");
    await card.click("Show less");
    expect(card.button("Show less")).toBeUndefined(); // five is the floor
  });

  it("collapses to its header, keeping the scope, and expands to the same rows", async () => {
    const card = runCard();
    await card.initialize();
    await card.open({}, things(1, 20), { total_items: 20, has_next: false, next_offset: null });
    await card.click("Show more");
    await card.click("Collapse");
    expect(card.rows()).toEqual([]);
    expect(card.scope()).toBe("All things · Showing 10 of 20");
    expect(card.button("Expand")!.attributes["aria-expanded"]).toBe("false");
    await card.click("Expand");
    expect(card.rows()).toHaveLength(10);
  });

  it("opens a row's detail in the card and returns to the same list", async () => {
    const card = runCard();
    await card.initialize();
    await card.open({}, things(1, 20), { total_items: 20, has_next: false, next_offset: null });
    await card.click("Show more");
    const openable = card.root.all((n) => n.tag === "button" && n.className === "row");
    expect(openable).toHaveLength(5); // the odd things are files; folders have no detail
    await openable[0]!.listeners.click!();
    expect(card.root.text).toContain("Kind");
    expect(card.root.text).toContain("2.0 KB");
    await card.click("Back");
    expect(card.rows()).toHaveLength(10);
  });

  it("keeps its rows when a page fails, and retries", async () => {
    const card = runCard();
    await card.initialize();
    await card.open({}, things(1, 5), { total_items: 9, has_next: true, next_offset: 5 });
    const failing = card.click("Show more");
    await card.deliver({ id: card.sent.at(-1)!.id, error: { message: "Galaxy is down" } });
    await failing;
    expect(card.rows()).toHaveLength(5);
    expect(card.root.text).toContain("Galaxy is down");
    const retrying = card.click("Retry");
    const call = card.sent.at(-1)!;
    expect(call.params.arguments.offset).toBe(5);
    await card.deliver({ id: call.id, result: result(things(6, 4), { total_items: 9, has_next: false, next_offset: null }) });
    await retrying;
    expect(card.rows()).toHaveLength(9);
    expect(card.button("Show more")).toBeUndefined();
  });

  it("steps five rows from a first page shorter than five", async () => {
    const card = runCard();
    await card.initialize();
    await card.open({ limit: 2 }, things(1, 2), { total_items: 9, has_next: true, next_offset: 2 });
    expect(card.button("Show less")).toBeUndefined();
    const showing = card.click("Show more");
    await card.deliver({ id: card.sent.at(-1)!.id, result: result(things(3, 7), { total_items: 9, has_next: false, next_offset: null }) });
    await showing;
    expect(card.rows()).toHaveLength(7);
  });

  it("offers Retry when the opening call failed, and loads with the same arguments", async () => {
    const card = runCard();
    await card.initialize();
    await card.deliver({ method: "ui/notifications/tool-input", params: { arguments: { filter: "big", limit: 8 } } });
    await card.deliver({ method: "ui/notifications/tool-result", params: { isError: true, content: [{ type: "text", text: "Galaxy is down" }] } });
    expect(card.scope()).toBe("Could not load");
    expect(card.root.text).toContain("Galaxy is down");
    const retrying = card.click("Retry");
    const call = card.sent.at(-1)!;
    expect(call.params).toEqual({ name: "get_things", arguments: { filter: "big", limit: 8 } });
    await card.deliver({ id: call.id, result: result(things(1, 8), { total_items: 8, has_next: false, next_offset: null }) });
    await retrying;
    expect(card.scope()).toBe("Filtered by big \u00b7 Showing 5 of 8");
  });

  it("shows the host's own words when a result is not Galaxy's, and offers Retry", async () => {
    const card = runCard();
    await card.initialize();
    await card.deliver({ method: "ui/notifications/tool-input", params: { arguments: {} } });
    await card.deliver({ method: "ui/notifications/tool-result", params: { content: [{ type: "text", text: "Tool result too large to show" }] } });
    expect(card.scope()).toBe("Could not load");
    expect(card.root.text).toContain("Tool result too large to show");
    expect(card.button("Retry")).toBeDefined();
  });

  it("offers full screen only where the host has it", async () => {
    const plain = runCard();
    await plain.initialize({ availableDisplayModes: ["inline"] });
    expect(plain.button("Full screen")).toBeUndefined();

    const card = runCard();
    await card.initialize({ availableDisplayModes: ["inline", "fullscreen"], displayMode: "inline" });
    const asking = card.click("Full screen");
    expect(card.sent.at(-1)).toMatchObject({ method: "ui/request-display-mode", params: { mode: "fullscreen" } });
    await card.deliver({ id: card.sent.at(-1)!.id, result: { mode: "fullscreen" } });
    await asking;
    expect(card.button("Exit full screen")).toBeDefined();
  });

  it("answers the host's teardown, even under an id its own request uses", async () => {
    const card = runCard();
    const ownId = card.sent[0]!.id;
    await card.deliver({ id: ownId, method: "ui/resource-teardown", params: {} });
    expect(card.sent.at(-1)).toEqual({ jsonrpc: "2.0", id: ownId, result: {} });
    await card.initialize();
    expect(card.sent.at(-2)?.method).toBe("ui/notifications/initialized");
  });

  it("is one self-contained page", () => {
    const html = cardPage(spec);
    expect(html).not.toMatch(/\b(src|href)=|@import|url\(/);
  });
});
