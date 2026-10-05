import { describe, it, expect } from "vitest";
import { applySectionEdit, contentHash, malformedObjectIds } from "../../src/operations/pages-common";

const REAL = "a8539f6d9115ffe7";
const ALSO_REAL = "0c97fda4aafcf418";
const galaxy = (directive: string) => "```galaxy\n" + directive + "\n```\n";

describe("contentHash", () => {
  it("is Galaxy's page hash", () => {
    expect(contentHash({ content_editor: "hello" })).toBe("0f923099");
  });

  it("walks code points, as page_assistant.py does", () => {
    expect(contentHash({ content_editor: "héllo😀" })).toBe("0b2edc3d");
  });
});

describe("applySectionEdit", () => {
  const DOC = "## Record\n\nintro\n\n## Methods\n\nold\n\n## Results\n\nfindings\n";

  it("leaves the other sections as they were", () => {
    expect(applySectionEdit(DOC, "## Methods", "## Methods\n\nnew\n")).toBe(
      "## Record\n\nintro\n\n## Methods\n\nnew\n\n## Results\n\nfindings\n",
    );
  });

  it("replaces the text before the first heading by the empty heading", () => {
    expect(applySectionEdit("preamble\n## A\n\na", "", "lead")).toBe("lead\n## A\n\na");
  });
});

describe("malformedObjectIds", () => {
  it("names an argument that holds something other than an encoded id", () => {
    expect(malformedObjectIds(galaxy("history_dataset_display(history_dataset_id=reads)"))).toEqual([
      "history_dataset_id=reads",
    ]);
  });

  it("takes an encoded id", () => {
    expect(malformedObjectIds(galaxy(`history_dataset_display(history_dataset_id=${REAL})`))).toEqual([]);
  });

  it("leaves a visualization's plugin name alone, which is a name and not an id", () => {
    expect(
      malformedObjectIds(galaxy(`visualization(visualization_id=plotly, history_dataset_id=${ALSO_REAL})`)),
    ).toEqual([]);
    expect(
      malformedObjectIds(galaxy("visualization(visualization_id=plotly, history_dataset_id=reads)")),
    ).toEqual(["history_dataset_id=reads"]);
  });

  it("leaves arguments Galaxy does not decode alone", () => {
    expect(malformedObjectIds(galaxy('history_dataset_display(output="trimmed reads", hid=3)'))).toEqual([]);
  });

  it("reads a quoted id through its quotes", () => {
    expect(malformedObjectIds(galaxy(`history_dataset_display(history_dataset_id="${REAL}")`))).toEqual([]);
  });

  it("reads nothing outside a directive", () => {
    expect(malformedObjectIds("## Record\n\nhistory_dataset_id=reads in a sentence.")).toEqual([]);
  });

  it.each(["reads", "0c97fda4aafcf41", "0c97fda4aafcf4188", "ZZZZZZZZZZZZZZZZ", "12"])(
    "refuses %s, which no cipher block produces",
    (value) => {
      expect(malformedObjectIds(galaxy(`history_dataset_display(history_dataset_id=${value})`))).not.toEqual([]);
    },
  );
});
