import { describe, expect, it } from "vitest";
import { cardPage } from "../src/apps/card-page";
import { datasetsCard, historiesCard, historyCard } from "../src/apps/cards";
import { runCard } from "./card-host";
import { answer, contents, histories, openings } from "./preview/sample";

const result = (tool: string, args: Record<string, unknown>) => ({ content: [{ type: "text", text: JSON.stringify(answer(tool, args)) }] });

describe("the preview's sample results", () => {
  it("page like the reads, so a previewed card can fetch its next page", () => {
    const first = answer("show_histories", { limit: 10 });
    expect(first.data).toHaveLength(10);
    expect(first.pagination).toMatchObject({ total_items: histories.length, has_next: true, next_offset: 10 });
    const rest = answer("get_histories", { limit: 20, offset: 10 });
    expect(rest.data).toHaveLength(histories.length - 10);
    expect(rest.pagination).toMatchObject({ has_next: false, next_offset: null });
  });

  it("leave deleted and hidden items out unless asked, and can run newest first", () => {
    const shown = (args: Record<string, unknown>) => (answer("get_history_contents", { history_id: "h1", limit: 100, ...args }).data as { contents: { hid: number }[] }).contents.map((item) => item.hid);
    expect(shown({})).not.toContain(16);
    expect(shown({})).not.toContain(17);
    expect(shown({ deleted: true })).toEqual([16]);
    expect(shown({ visible: false })).toEqual([17]);
    expect(shown({ order: "hid-dsc" })[0]).toBe(24);
    expect(answer("get_history_contents", { history_id: "elsewhere" }).success).toBe(false);
  });

  it("cover what a card treats differently: both kinds of item, a failure, and names that look like markup", () => {
    expect(new Set(contents.map((item) => item.history_content_type))).toEqual(new Set(["dataset", "dataset_collection"]));
    const states = contents.flatMap((item) => ("state" in item ? [item.state] : []));
    for (const state of ["ok", "error", "running", "queued", "new", "paused", "upload"]) expect(states, state).toContain(state);
    expect(contents.some((item) => item.name.includes("<img"))).toBe(true);
    expect(histories.some((history) => history.name.includes("<b>"))).toBe(true);
  });

  it("open each card as a host would, with every kind of selected dataset", async () => {
    for (const [name, card] of Object.entries({ histories: historiesCard, history: historyCard, datasets: datasetsCard })) {
      const { tool, args } = openings[name]!;
      const shown = runCard(cardPage(card));
      await shown.initialize();
      await shown.deliver({ method: "ui/notifications/tool-input", params: { arguments: args } });
      await shown.deliver({ method: "ui/notifications/tool-result", params: result(tool, args) });
      expect(shown.rows().length, name).toBeGreaterThan(0);
    }
    const selected = (answer("show_datasets", openings.datasets!.args).data as { selected: { status: string }[] }).selected;
    expect(new Set(selected.map((entry) => entry.status))).toEqual(new Set(["available", "unavailable", "inaccessible"]));
  });
});
