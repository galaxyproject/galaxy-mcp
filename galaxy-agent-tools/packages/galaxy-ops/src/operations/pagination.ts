/**
 * Page windows for the list-returning ops.
 *
 * An MCP client truncates a large tool result -- one common adapter cuts at
 * 50 KB and what the model then sees is no longer JSON -- so every listing
 * returns a window and says where it sits.
 *
 * Two things keep the window from overrunning that limit, and they are the same
 * two the Python server uses. A ceiling refuses a request nobody should make:
 * each op's is the number Python's MAX_PAGE_SIZE gives that tool, so a call one
 * surface accepts the other accepts too, and the ops Python does not cap are not
 * capped here either. A count cannot promise a size, though -- Galaxy names,
 * descriptions and readmes have no maximum -- so the result is then measured as
 * the text a model reads and the page is cut until it fits. The ceiling is about
 * the request; the budget is about the answer.
 */
import { GalaxyValidationError } from "../errors";
import type { Pagination } from "./types";

/** A fully described page window. */
export interface PaginationInfo extends Pagination {
  /** How many items exist in total, across every page. */
  total: number;
  /** How many items this page actually carries. */
  returned: number;
  limit: number;
  offset: number;
  hasNext: boolean;
  hasPrevious: boolean;
  nextOffset?: number;
  previousOffset?: number;
  helperText: string;
}

/** What an op's `run` returns once it pages: the page, and where the page sits. */
export interface Paged<T> {
  items: T[];
  pagination: PaginationInfo;
}

/**
 * The budget a single tool result has to fit: the same 50,000 bytes the Python
 * server measures against, so a page one surface considers safe is safe on the
 * other. Bytes, not characters -- a name in a non-Latin script is several bytes
 * per character and a length in characters can be half the truth.
 */
export const OUTPUT_BUDGET_BYTES = 50_000;

/**
 * What to tell a caller that asked for a bigger page than the tool allows.
 *
 * The sentence the Python server uses, word for word and in its order, because the
 * two surfaces refuse the same window and an agent should not have to notice which
 * one it is talking to. `pageable` is false for a listing with no offset, which
 * cannot act on advice to page through the rest.
 */
export const overCapMessage = (maxLimit: number, got: number, pageable = true): string =>
  `limit must be at most ${maxLimit} (got ${got}); request ${maxLimit} or fewer` +
  (pageable ? " and use offset to page through the rest" : "");

/**
 * Reject a window nobody should be asking for.
 *
 * `maxLimit` is left out for the ops Python does not validate, which then take
 * any positive limit, as they do there. Thrown before any network call so a bad
 * window fails the same way every time rather than depending on whether Galaxy is
 * reachable.
 */
export function validatePagination(
  limit: number,
  offset: number,
  opts: { maxLimit?: number; pageable?: boolean } = {},
): void {
  if (!Number.isInteger(limit)) {
    throw new GalaxyValidationError(`limit must be a whole number (got ${limit})`);
  }
  if (limit < 1) {
    throw new GalaxyValidationError(`limit must be at least 1 (got ${limit})`);
  }
  if (opts.maxLimit !== undefined && limit > opts.maxLimit) {
    throw new GalaxyValidationError(overCapMessage(opts.maxLimit, limit, opts.pageable ?? true));
  }
  if (!Number.isInteger(offset) || offset < 0) {
    throw new GalaxyValidationError(`offset must be 0 or greater (got ${offset})`);
  }
}

/**
 * Describe one page of `total` results.
 *
 * `hasNext` comes from what was actually returned rather than from `limit`, so
 * a short final page reports itself as the last one.
 */
export function paginationInfo(opts: {
  total: number;
  returned: number;
  limit: number;
  offset: number;
  noun: string;
  /** The page was cut to fit the output budget, not because there is nothing more. */
  trimmedForSize?: boolean;
}): PaginationInfo {
  const { returned, limit, offset, noun } = opts;
  // An op that windows server-side reads its total separately, so the two can
  // disagree. Items in hand prove a floor; an empty page proves nothing, so an
  // offset past the end must not inflate the total.
  const total = returned > 0 ? Math.max(opts.total, offset + returned) : opts.total;
  const hasNext = offset + returned < total;
  const hasPrevious = offset > 0;
  // Advance by what we got, falling back to limit so an empty page never stalls
  // a page-walking caller on the same offset.
  const nextOffset = offset + (returned || limit);

  // A cut page is short for a reason that is not "there is nothing more", and an
  // agent reading a short page would otherwise conclude exactly that.
  const cut = opts.trimmedForSize
    ? " This page was cut short to fit the output budget, not because there is nothing more."
    : "";
  const helperText =
    returned === 0 && total > 0 && offset >= total
      ? `offset ${offset} is past the end of ${total} ${noun}; use a smaller offset`
      : `Showing ${returned} of ${total} ${noun} (offset ${offset}).${cut} ` +
        (hasNext ? `Use offset=${nextOffset} for the next page.` : "This is the last page.");

  return {
    total,
    returned,
    limit,
    offset,
    hasNext,
    hasPrevious,
    ...(hasNext ? { nextOffset } : {}),
    ...(hasPrevious ? { previousOffset: Math.max(0, offset - limit) } : {}),
    ...(opts.trimmedForSize ? { trimmedForSize: true } : {}),
    helperText,
  };
}

/** Slice `items` client-side and describe the window. */
export function paginate<T>(
  items: readonly T[],
  opts: { limit: number; offset: number; noun: string },
): Paged<T> {
  const page = items.slice(opts.offset, opts.offset + opts.limit);
  return {
    items: page,
    pagination: paginationInfo({
      total: items.length,
      returned: page.length,
      limit: opts.limit,
      offset: opts.offset,
      noun: opts.noun,
    }),
  };
}

/**
 * Cut a result's page until the text a model reads fits the output budget.
 *
 * The port of the Python server's `_budgeted_page`: measure the whole serialised
 * result, scale the page by how far over it is, and never return nothing -- one
 * item over budget by itself is returned anyway, because a list op has nothing
 * better to offer and trimming fields inside the item would hand back something
 * that is not the item.
 *
 * `rows` and `shrink` come from the op, which is the only thing that knows where
 * its rows live and how to describe a shorter page. `render` produces exactly the
 * text the surface will emit, so the number measured here is the number that
 * reaches the other end.
 */
export function trimToBudget<O>(
  data: O,
  budget: { rows(data: O): number; shrink(data: O, keep: number): O },
  render: (data: O) => string,
): O {
  const encoder = new TextEncoder();
  let page = data;
  for (;;) {
    const size = encoder.encode(render(page)).length;
    const rows = budget.rows(page);
    if (size <= OUTPUT_BUDGET_BYTES || rows <= 1) return page;
    // Scale by how far over budget we are, and always drop at least one row so
    // this cannot stall on a page whose overshoot is all envelope.
    page = budget.shrink(page, Math.min(rows - 1, Math.max(1, Math.floor((rows * OUTPUT_BUDGET_BYTES) / size))));
  }
}

/**
 * A `Paged` result cut to `keep` rows, saying it was cut.
 *
 * What the `budget.shrink` of every op that returns a plain page is, so the eight
 * of them do not each write the same four lines.
 */
export function shrinkPaged<T>(page: Paged<T>, keep: number, noun: string): Paged<T> {
  return {
    items: page.items.slice(0, keep),
    pagination: paginationInfo({
      total: page.pagination.total,
      returned: keep,
      limit: page.pagination.limit,
      offset: page.pagination.offset,
      noun,
      trimmedForSize: true,
    }),
  };
}
