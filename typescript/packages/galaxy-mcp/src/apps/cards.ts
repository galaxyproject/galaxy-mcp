/**
 * The cards and the read each one pages through. Everything here past the URIs is
 * page-side JavaScript: it runs in the host's frame, against the result of a tool call.
 */
import type { CardSpec } from "./card-page.js";

export const HISTORIES_CARD_URI = "ui://galaxy/histories.html";
export const HISTORY_CARD_URI = "ui://galaxy/history.html";
export const DATASETS_CARD_URI = "ui://galaxy/datasets.html";

// A page of a paged read: its rows, Galaxy's total and where the next page starts.
const paged = (items: string) => `(page) => ({
  items: ${items} || [],
  total: typeof page.pagination?.total_items === "number" ? page.pagination.total_items : null,
  nextOffset: page.pagination?.has_next ? page.pagination.next_offset : null,
})`;

const datasetDetail = `[
  ["Format", item.extension], ["State", item.state], ["Size", bytes(item.file_size)],
  ["Created", galaxyTime(item.create_time)], ["Updated", galaxyTime(item.update_time)],
  ["Deleted", item.deleted ? "yes" : ""], ["Hidden", item.visible === false ? "yes" : ""], ["ID", item.id],
]`;

export const historiesCard: CardSpec = {
  uri: HISTORIES_CARD_URI,
  title: "Galaxy histories",
  pageTool: "get_histories",
  openTool: "get_histories",
  keep: ["name"],
  read: `(page, kept) => ({
    ...(${paged("page.data")})(page),
    scope: kept.name !== undefined && kept.name !== null ? "Named \\u201c" + kept.name + "\\u201d" : "Your histories",
  })`,
  row: `(history) => ({
    name: history.name || "Unnamed history",
    meta: [
      typeof history.count === "number" ? history.count + (history.count === 1 ? " item" : " items") : "",
      galaxyTime(history.update_time) && "updated " + galaxyTime(history.update_time),
    ],
  })`,
  empty: "No histories yet.",
};

export const historyCard: CardSpec = {
  uri: HISTORY_CARD_URI,
  title: "Galaxy history",
  pageTool: "get_history_contents",
  openTool: "get_history_contents",
  keep: ["history_id", "deleted", "visible", "order"],
  read: `(page, kept) => ({
    ...(${paged("page.data?.contents")})(page),
    // get_history_contents leaves deleted and hidden items out unless asked for them.
    scope: ["Browsing history",
      [kept.deleted ? "" : "deleted", kept.visible === false ? "" : "hidden"].filter(Boolean).join(" or ").replace(/^(.+)$/, "without $1 items"),
      kept.order && kept.order !== "hid-asc" ? "sorted by " + kept.order : "",
    ].filter(Boolean).join(", "),
  })`,
  row: `(item) => item.history_content_type === "dataset_collection"
    ? {
        name: item.hid + ": " + (item.name || "Unnamed collection"),
        meta: [(item.collection_type || "collection") + (typeof item.element_count === "number" ? ", " + item.element_count + " elements" : "")],
        state: item.populated_state && item.populated_state !== "ok" ? item.populated_state : "",
      }
    : {
        name: item.hid + ": " + (item.name || "Unnamed dataset"),
        meta: [item.extension, item.deleted ? "deleted" : "", item.visible === false ? "hidden" : ""],
        state: item.state,
        detail: ${datasetDetail},
      }`,
  empty: "Nothing in this history matches.",
};

export const datasetsCard: CardSpec = {
  uri: DATASETS_CARD_URI,
  title: "Selected datasets",
  openTool: "show_datasets",
  keep: [],
  read: `(page) => {
    const selected = page.data?.selected || [];
    const missing = selected.length - (page.data?.available ?? 0);
    return {
      items: selected,
      total: selected.length,
      nextOffset: null,
      caption: page.data?.caption || "",
      scope: ["Chosen by the assistant", missing ? missing + " unavailable" : ""].filter(Boolean).join(" \\u00b7 "),
    };
  }`,
  row: `(entry) => {
    const item = entry.dataset || {};
    const name = item.hid !== undefined ? item.hid + ": " + (item.name || "Unnamed dataset") : entry.id;
    if (entry.status === "unavailable") return { name, meta: ["Not available in this history"], state: "" };
    if (entry.status === "inaccessible") return { name, meta: ["You do not have access to this dataset"], state: "" };
    return { name: name + (item.name_truncated ? "\\u2026" : ""), meta: [item.extension, bytes(item.file_size)], state: item.state, detail: ${datasetDetail} };
  }`,
  empty: "No datasets were selected.",
};
