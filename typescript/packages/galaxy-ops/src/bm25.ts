import { codePointBefore, isPyWordChar } from "./python-str";

/** The other server's `stop_words`, word for word. */
const STOPWORDS = new Set([
  "the", "and", "for", "with", "from", "have", "want",
  "data", "this", "that", "are", "was", "will",
]);

const isAsciiLetter = (code: number): boolean =>
  (code >= 0x41 && code <= 0x5a) || (code >= 0x61 && code <= 0x7a);

/**
 * `_tokenize_for_search`: runs of two or more ASCII letters standing alone as a word,
 * lowercased, minus thirteen stop words.
 *
 * The other server writes it as `re.findall(r"\b[a-zA-Z]{2,}\b", text)`, and the two halves of
 * that pattern do not travel the same way. `[a-zA-Z]{2,}` is ASCII and means the same thing in
 * both languages -- a token is Latin letters, so "RNA" is a term and so is "seq", while
 * "bwa2", "2bwa" and a CJK word are not terms at all. `\b` is not: it is a change of side
 * across whatever the engine calls a word character, `re` calls every letter and digit in
 * Unicode one, and JavaScript calls only `[A-Za-z0-9_]` one. Left as `\b`, an intent of "café"
 * tokenises to nothing there and to `caf` here, and the difference is not a ranking detail --
 * nothing to tokenise is one of that tool's early returns, so one surface says "No searchable
 * terms in query" and the other says it found no matches, which sends an agent to rewrite a
 * query that was never the problem.
 *
 * So the boundary is drawn by hand instead: find each run of ASCII letters and keep it when
 * the code points on either side of it are not word characters, which is exactly what a `\b`
 * at each end of the run asks. `isPyWordChar` answers from the contract interpreter's table
 * rather than from this runtime's `\p{L}`, so a letter Unicode assigned after that
 * interpreter's edition -- U+A7CB, say -- is a separator on both sides, and "rnaseq" beside
 * one is a term on both sides too.
 *
 * One pass, no regex: the scan walks code units, and only the two neighbours of a candidate
 * run are decoded as code points, so half of an astral letter is never read as a separator.
 * Lowercasing a run of ASCII letters is the same operation in both languages, so `toLowerCase`
 * is `str.lower()` here without any of the Unicode data question the boundary had.
 *
 * The corpus is tokenised with this too, so the rule moves both sides of the match together:
 * a readme saying "café" indexes no `caf` term any more, and an intent that asked for one
 * stops matching it. That is the other server's ranking, which is the point.
 */
export function tokenizeForSearch(text: string): string[] {
  const tokens: string[] = [];
  let index = 0;
  while (index < text.length) {
    if (!isAsciiLetter(text.charCodeAt(index))) {
      index += 1;
      continue;
    }
    const start = index;
    while (index < text.length && isAsciiLetter(text.charCodeAt(index))) index += 1;
    if (index - start < 2) continue;
    const before = codePointBefore(text, start);
    const after = text.codePointAt(index);
    if (before !== null && isPyWordChar(before)) continue;
    if (after !== undefined && isPyWordChar(after)) continue;
    const token = text.slice(start, index).toLowerCase();
    if (!STOPWORDS.has(token)) tokens.push(token);
  }
  return tokens;
}

/**
 * BM25Okapi implementation matching rank_bm25.BM25Okapi defaults.
 * k1=1.5, b=0.75, epsilon=0.25
 */
export class BM25Okapi {
  private readonly k1 = 1.5;
  private readonly b = 0.75;
  private readonly epsilon = 0.25;

  private readonly N: number;
  private readonly avgdl: number;
  private readonly docFreqs: Array<Map<string, number>>;
  private readonly docLens: number[];
  private readonly idf: Map<string, number>;

  constructor(corpus: string[][]) {
    this.N = corpus.length;
    this.docFreqs = [];
    this.docLens = [];

    const nd = new Map<string, number>(); // word -> number of docs containing it

    let totalLen = 0;
    for (const doc of corpus) {
      const tf = new Map<string, number>();
      for (const word of doc) {
        tf.set(word, (tf.get(word) ?? 0) + 1);
      }
      this.docFreqs.push(tf);
      this.docLens.push(doc.length);
      totalLen += doc.length;

      for (const word of tf.keys()) {
        nd.set(word, (nd.get(word) ?? 0) + 1);
      }
    }

    this.avgdl = this.N > 0 ? totalLen / this.N : 0;
    this.idf = this._computeIdf(nd);
  }

  private _computeIdf(nd: Map<string, number>): Map<string, number> {
    const idfMap = new Map<string, number>();
    if (this.N === 0) return idfMap;

    let idfSum = 0;
    const negatives: string[] = [];

    for (const [word, freq] of nd.entries()) {
      const idf = Math.log(this.N - freq + 0.5) - Math.log(freq + 0.5);
      idfMap.set(word, idf);
      idfSum += idf;
      if (idf < 0) negatives.push(word);
    }

    const numWords = nd.size;
    const averageIdf = numWords > 0 ? idfSum / numWords : 0;
    const eps = this.epsilon * averageIdf;

    for (const word of negatives) {
      idfMap.set(word, eps);
    }

    return idfMap;
  }

  getScores(query: string[]): number[] {
    if (this.N === 0) return [];

    const scores = new Array<number>(this.N).fill(0);

    for (const q of query) {
      const idf = this.idf.get(q);
      if (idf === undefined) continue;

      for (let i = 0; i < this.N; i++) {
        const docTf = this.docFreqs[i];
        const dl = this.docLens[i];
        if (docTf === undefined || dl === undefined) continue;
        const f = docTf.get(q) ?? 0;
        if (f === 0) continue;
        const denom = f + this.k1 * (1 - this.b + this.b * dl / (this.avgdl || 1));
        const slot = scores[i];
        if (slot !== undefined) {
          scores[i] = slot + idf * (f * (this.k1 + 1)) / denom;
        }
      }
    }

    return scores;
  }
}
