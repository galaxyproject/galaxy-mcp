import { describe, it, expect } from "vitest";
import { tokenizeForSearch, BM25Okapi } from "../src/bm25";

describe("tokenizeForSearch", () => {
  it("drops stopwords, 1-char tokens, and digits; lowercases; keeps order", () => {
    expect(tokenizeForSearch("RNA-seq data with 2 reads")).toEqual(["rna", "seq", "reads"]);
  });

  it("drops single-char tokens", () => {
    expect(tokenizeForSearch("a b c hello")).toEqual(["hello"]);
  });

  it("lowercases all tokens", () => {
    expect(tokenizeForSearch("BLAST Alignment")).toEqual(["blast", "alignment"]);
  });

  it("drops stopwords", () => {
    expect(tokenizeForSearch("the data are from the source")).toEqual(["source"]);
  });

  it("returns empty array for empty string", () => {
    expect(tokenizeForSearch("")).toEqual([]);
  });

  it("returns empty array for stopwords-only input", () => {
    expect(tokenizeForSearch("the and for with from")).toEqual([]);
  });
});

/**
 * The same text, tokenised by the other server.
 *
 * Every expected list below came off the installed interpreter rather than out of a
 * reading of the pattern, from this script:
 *
 *   import re
 *   stop_words = {"the","and","for","with","from","have","want",
 *                 "data","this","that","are","was","will"}
 *   def tok(text):
 *       return [w.lower() for w in re.findall(r"\b[a-zA-Z]{2,}\b", text)
 *               if w.lower() not in stop_words]
 *
 * which is `_tokenize_for_search` with its imports inlined. The cases are the ones where
 * `\b` decides the answer, because that is the half of the pattern JavaScript spells
 * differently: `re` counts every Unicode letter and digit as a word character, so a run of
 * ASCII letters touching one is not a word, and an ASCII `\b` would hand back the Latin
 * prefix of it.
 */
describe("tokenizeForSearch draws word boundaries where the other server draws them", () => {
  const cases: Array<[string, string[]]> = [
    // 'café' -> []
    ["café", []],
    // 'café au lait' -> ['au', 'lait']
    ["café au lait", ["au", "lait"]],
    // 'naïve RNA-seq' -> ['rna', 'seq']
    ["naïve RNA-seq", ["rna", "seq"]],
    // '!!! --- ???' -> []
    ["!!! --- ???", []],
    // '測序 分析' -> []
    ["測序 分析", []],
    // 'bwa2' -> []
    ["bwa2", []],
    // '2bwa' -> []
    ["2bwa", []],
    // 'bwa 2' -> ['bwa']
    ["bwa 2", ["bwa"]],
    // 'RNA-Seq DATA Analysis' -> ['rna', 'seq', 'analysis']
    ["RNA-Seq DATA Analysis", ["rna", "seq", "analysis"]],
    // 'rna²' -> []
    ["rna²", []],
    // 'RNA_seq' -> []
    ["RNA_seq", []],
    // 'İstanbul' -> []
    ["İstanbul", []],
    // 'straße rnaseq' -> ['rnaseq']
    ["straße rnaseq", ["rnaseq"]],
    // 'α-helix alignment' -> ['helix', 'alignment']
    ["α-helix alignment", ["helix", "alignment"]],
  ];

  for (const [text, expected] of cases) {
    it(`tokenises ${JSON.stringify(text)} to ${JSON.stringify(expected)}`, () => {
      expect(tokenizeForSearch(text)).toEqual(expected);
    });
  }

  it("reads a letter outside the Basic Multilingual Plane as one character, not two", () => {
    // 'rna𝐚' is one letter after the three, so there is no word here at all -- and a
    // boundary test without the u flag would inspect the high surrogate, find it in no
    // class, and call 'rna' a word.
    expect(tokenizeForSearch("rna\u{1D41A}")).toEqual([]);
    expect(tokenizeForSearch("rna \u{1D41A}")).toEqual(["rna"]);
  });
});

describe("BM25Okapi", () => {
  it("returns empty array when corpus is empty", () => {
    const bm25 = new BM25Okapi([]);
    expect(bm25.getScores(["cat"])).toEqual([]);
  });

  it("a doc with repeated query term scores higher than one with it once", () => {
    const corpus = [
      ["apple", "apple", "apple"],
      ["apple", "banana"],
      ["cherry"],
    ];
    const bm25 = new BM25Okapi(corpus);
    const scores = bm25.getScores(["apple"]);
    expect(scores[0]).toBeGreaterThan(scores[1]);
    expect(scores[1]).toBeGreaterThan(0);
  });

  it("a doc with no query terms scores 0", () => {
    // 3 docs: apple in 1/3, giving positive IDF; cherry has no apple
    const corpus = [
      ["apple", "banana"],
      ["cherry", "date"],
      ["elderberry", "fig"],
    ];
    const bm25 = new BM25Okapi(corpus);
    const scores = bm25.getScores(["apple"]);
    expect(scores[0]).toBeGreaterThan(0);
    expect(scores[1]).toBe(0);
    expect(scores[2]).toBe(0);
  });

  it("ordering is stable and deterministic", () => {
    const corpus = [
      ["rna", "seq", "alignment"],
      ["rna", "rna", "rna", "seq"],
      ["proteomics", "mass", "spectrometry"],
    ];
    const bm25 = new BM25Okapi(corpus);
    const scores1 = bm25.getScores(["rna", "seq"]);
    const scores2 = bm25.getScores(["rna", "seq"]);
    expect(scores1).toEqual(scores2);
    // docs 0 and 1 score above 0; doc 2 scores 0
    expect(scores1[0]).toBeGreaterThan(0);
    expect(scores1[1]).toBeGreaterThan(0);
    expect(scores1[2]).toBe(0);
  });

  it("golden numeric assertion: 3-doc apple corpus", () => {
    // corpus = [["apple","apple","apple"],["apple","banana"],["cherry"]]
    // N=3, nd={apple:2, banana:1, cherry:1}
    // idf(apple) = log(1.5)-log(2.5) ≈ -0.5108 (negative)
    // idf(banana) = log(2.5)-log(1.5) ≈ 0.5108
    // idf(cherry) ≈ 0.5108
    // idfSum = -0.5108+0.5108+0.5108 = 0.5108
    // averageIdf = 0.5108/3 ≈ 0.17027
    // eps = 0.25 * 0.17027 ≈ 0.04257
    // idfMap[apple] = eps ≈ 0.04257
    // avgdl = (3+2+1)/3 = 2
    // doc0: f=3, docLen=3
    //   denom = 3 + 1.5*(0.25 + 0.75*3/2) = 3 + 1.5*1.375 = 5.0625
    //   score = 0.04257 * (3*2.5)/5.0625 ≈ 0.04257 * 1.4815 ≈ 0.06306
    // doc1: f=1, docLen=2
    //   denom = 1 + 1.5*(0.25 + 0.75*1) = 1 + 1.5 = 2.5
    //   score = 0.04257 * (1*2.5)/2.5 = 0.04257
    const corpus = [["apple", "apple", "apple"], ["apple", "banana"], ["cherry"]];
    const bm25 = new BM25Okapi(corpus);
    const scores = bm25.getScores(["apple"]);
    expect(scores).toHaveLength(3);
    expect(scores[0]).toBeCloseTo(0.06306, 3);
    expect(scores[1]).toBeCloseTo(0.04257, 3);
    expect(scores[2]).toBe(0);
    // ordering: doc0 > doc1 > doc2
    expect(scores[0]).toBeGreaterThan(scores[1]);
    expect(scores[1]).toBeGreaterThan(scores[2]);
  });
});
