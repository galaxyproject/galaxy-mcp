import { describe, it, expect } from "vitest";
import { applySectionEdit, malformedObjectIds } from "../../src/operations/pages-common";

const DOC = "## Record\n\nintro\n\n## Methods\n\nold\n\n## Results\n\nfindings\n";
const galaxy = (directive: string) => "```galaxy\n" + directive + "\n```\n";

describe("applySectionEdit", () => {
  it.each([
    ["## Methods", "## Methods\n\nnew\n", "## Record\n\nintro\n\n## Methods\n\nnew\n\n## Results\n\nfindings\n"],
    ["## Notes", "## Notes\n\nadded", DOC + "\n## Notes\n\nadded"],
  ])("replaces or appends %s", (heading, section, edited) => {
    expect(applySectionEdit(DOC, heading, section)).toBe(edited);
  });

  it("replaces the text before the first heading by the empty heading", () => {
    expect(applySectionEdit("preamble\n## A\n\na", "", "lead")).toBe("lead\n## A\n\na");
  });
});

describe("malformedObjectIds", () => {
  it.each([
    ["an encoded id", galaxy("history_dataset_display(history_dataset_id=a8539f6d9115ffe7)")],
    ["a quoted encoded id", galaxy('history_dataset_display(history_dataset_id="a8539f6d9115ffe7")')],
    ["a longer cipher output", galaxy(`invocation_outputs(invocation_id=${"0c97fda4aafcf418".repeat(2)})`)],
    ["a visualization's plugin name", galaxy("visualization(visualization_id=plotly, history_dataset_id=0c97fda4aafcf418)")],
    ["arguments Galaxy does not decode", galaxy('history_dataset_display(output="trimmed reads", hid=3)')],
    ["text outside a directive", "## Record\n\nhistory_dataset_id=reads in a sentence."],
  ])("accepts %s", (_, content) => {
    expect(malformedObjectIds(content)).toEqual([]);
  });

  it.each(["reads", "0c97fda4aafcf41", "0c97fda4aafcf4188", "ZZZZZZZZZZZZZZZZ", "12"])(
    "refuses %s, which no cipher block produces",
    (value) => {
      const content = galaxy(`visualization(visualization_id=plotly, history_dataset_id=${value})`);
      expect(malformedObjectIds(content)).toEqual([`history_dataset_id=${value}`]);
    },
  );
});
