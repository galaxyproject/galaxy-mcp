/**
 * Both ends of a log, cut on line boundaries, with a line saying what was left out.
 *
 * server.py's `_log_ends`, byte for byte, so change both. The cause of a failure is usually
 * at the end and the context at the start, and a one-ended read loses one of them. Measured
 * in UTF-8 bytes: `budget` is the most of the field's own bytes kept (the omitted line is
 * not charged against it), and 0 means uncut. Each half is cut back to its last (or forward
 * to its first) newline; a half with no newline is cut on a character boundary instead,
 * never inside a multi-byte character, which is what lets both decodes below be fatal.
 */
export function logEnds(text: string, budget: number): string {
  if (budget === 0) return text;
  // A lone surrogate becomes EF BF BD here; the Python side substitutes U+FFFD before it
  // encodes, so the two count the same bytes.
  const data = new TextEncoder().encode(text);
  if (data.length <= budget) return text;
  const half = Math.floor(budget / 2);
  const front = data.subarray(0, half);
  const cut = front.lastIndexOf(10);
  let head: Uint8Array;
  if (cut >= 0) {
    head = front.subarray(0, cut);
  } else {
    // The byte after the cut is a continuation byte only if the cut is inside a character;
    // walk back to that character's lead byte and leave it out.
    let end = half;
    while (end > 0 && (data[end]! & 0xc0) === 0x80) end -= 1;
    head = data.subarray(0, end);
  }
  const back = data.subarray(data.length - half);
  const newline = back.indexOf(10);
  let tail: Uint8Array;
  if (newline >= 0) {
    tail = back.subarray(newline + 1);
  } else {
    let start = data.length - half;
    while (start < data.length && (data[start]! & 0xc0) === 0x80) start += 1;
    tail = data.subarray(start);
  }
  const dropped = data.length - head.length - tail.length;
  // ignoreBOM is load-bearing: without it a tail that starts with U+FEFF loses it, where
  // Python's codec keeps it, and the two outputs differ by three bytes.
  const decoder = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
  return `${decoder.decode(head)}\n[... ${dropped} of ${data.length} bytes omitted ...]\n${decoder.decode(tail)}`;
}
