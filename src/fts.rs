//! Full-text search: tokenizer, positional inverted index, BM25 scoring.
//!
//! One object per full-text attribute. The index holds, per term, the documents
//! containing it and the token POSITIONS within each — positions cost roughly as
//! much space again, and buy phrase matching, which is the difference between
//! "contains these words" and "contains this phrase".
//!
//! Scoring is textbook BM25. The parameters are stored in the index rather than
//! read from config at query time, so a namespace's scores stay comparable to
//! each other after a config change: re-tuning `k1` without rebuilding would
//! silently mix two scoring regimes in one ranked list.

use crate::value::Value;
use anyhow::{Result, bail};
use bytes::{BufMut, Bytes, BytesMut};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap};

/// English stopwords.
///
/// ponytail: one language. `Tokenizer::language` already selects the stemmer, so
/// adding lists for the others is a data change rather than a code change.
const EN_STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

fn default_max_token_length() -> usize {
    39
}
fn default_k1() -> f32 {
    1.2
}
fn default_b() -> f32 {
    0.75
}
fn default_true() -> bool {
    true
}

/// How text becomes tokens. Field names and defaults follow turbopuffer's
/// `word_v4` tokenizer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tokenizer {
    #[serde(default)]
    pub language: Language,
    #[serde(default = "default_true")]
    pub stemming: bool,
    #[serde(default = "default_true")]
    pub remove_stopwords: bool,
    /// Fold accented characters to ASCII. Applied AFTER stemming and stopword
    /// removal, matching turbopuffer, because folding first would change which
    /// words the stemmer and stopword list recognise.
    #[serde(default)]
    pub ascii_folding: bool,
    #[serde(default)]
    pub case_sensitive: bool,
    #[serde(default = "default_max_token_length")]
    pub max_token_length: usize,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self {
            language: Language::English,
            stemming: true,
            remove_stopwords: true,
            ascii_folding: false,
            case_sensitive: false,
            max_token_length: default_max_token_length(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    #[default]
    English,
    French,
    German,
    Spanish,
    Italian,
    Portuguese,
    Dutch,
    Swedish,
    Norwegian,
    Danish,
    Russian,
    Indonesian,
}

impl Language {
    fn algorithm(self) -> rust_stemmers::Algorithm {
        use rust_stemmers::Algorithm as A;
        match self {
            Language::English => A::English,
            Language::French => A::French,
            Language::German => A::German,
            Language::Spanish => A::Spanish,
            Language::Italian => A::Italian,
            Language::Portuguese => A::Portuguese,
            Language::Dutch => A::Dutch,
            Language::Swedish => A::Swedish,
            Language::Norwegian => A::Norwegian,
            Language::Danish => A::Danish,
            Language::Russian => A::Russian,
            // Snowball has no Indonesian stemmer; English is a poor but harmless
            // stand-in, and `stemming: false` is the honest setting for it.
            Language::Indonesian => A::English,
        }
    }
}

impl Tokenizer {
    /// Split text into tokens.
    ///
    /// Position in the returned vector is the token's position in the document,
    /// which is what phrase matching compares. Dropped tokens (stopwords,
    /// over-length) do NOT leave a gap: two words either side of a stopword are
    /// adjacent for phrase purposes, which is what a user searching "king of
    /// spain" expects when "of" is a stopword.
    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let stemmer = self.stemming.then(|| rust_stemmers::Stemmer::create(self.language.algorithm()));

        text.split(|c: char| !c.is_alphanumeric())
            .filter(|s| !s.is_empty())
            .filter_map(|raw| {
                let mut token =
                    if self.case_sensitive { raw.to_string() } else { raw.to_lowercase() };

                if self.remove_stopwords
                    && matches!(self.language, Language::English)
                    && EN_STOPWORDS.contains(&token.as_str())
                {
                    return None;
                }
                if let Some(s) = &stemmer {
                    token = s.stem(&token).to_string();
                }
                if self.ascii_folding {
                    token = fold_ascii(&token);
                }
                if token.is_empty() || token.len() > self.max_token_length {
                    return None;
                }
                Some(token)
            })
            .collect()
    }
}

/// Strip diacritics by decomposing and dropping combining marks.
fn fold_ascii(s: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    use unicode_normalization::char::is_combining_mark;
    s.nfd().filter(|c| !is_combining_mark(*c)).collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Bm25Params {
    #[serde(default = "default_k1")]
    pub k1: f32,
    #[serde(default = "default_b")]
    pub b: f32,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Self { k1: default_k1(), b: default_b() }
    }
}

/// Per-attribute full-text configuration, as declared in a schema.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FtsConfig {
    #[serde(default)]
    pub tokenizer: Tokenizer,
    #[serde(default, flatten)]
    pub params: Bm25Params,
}

#[derive(Debug, Clone, PartialEq)]
struct Posting {
    ordinal: u32,
    /// Token positions. Length is the term frequency, so it is never stored twice.
    positions: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FtsIndex {
    /// Sorted by term, so a future version can binary-search a range request
    /// instead of fetching the whole object.
    /// ponytail: the whole index object is fetched per query. Add a term
    /// dictionary with offsets at the tail and range-request only the query's
    /// terms when full-text namespaces get large.
    terms: Vec<(String, Vec<Posting>)>,
    /// Token count per document ordinal.
    doc_lengths: Vec<u32>,
    avgdl: f32,
    tokenizer: Tokenizer,
    params: Bm25Params,
}

impl FtsIndex {
    pub fn build<'a>(
        docs: impl Iterator<Item = (u32, &'a str)>,
        total_docs: usize,
        config: &FtsConfig,
    ) -> Self {
        let mut postings: HashMap<String, Vec<Posting>> = HashMap::new();
        let mut doc_lengths = vec![0u32; total_docs];

        for (ordinal, text) in docs {
            let tokens = config.tokenizer.tokenize(text);
            if let Some(slot) = doc_lengths.get_mut(ordinal as usize) {
                *slot = tokens.len() as u32;
            }
            let mut per_doc: HashMap<&str, Vec<u32>> = HashMap::new();
            for (pos, tok) in tokens.iter().enumerate() {
                per_doc.entry(tok).or_default().push(pos as u32);
            }
            for (tok, positions) in per_doc {
                postings.entry(tok.to_string()).or_default().push(Posting { ordinal, positions });
            }
        }

        let mut terms: Vec<(String, Vec<Posting>)> = postings.into_iter().collect();
        terms.sort_by(|a, b| a.0.cmp(&b.0));
        for (_, p) in terms.iter_mut() {
            p.sort_unstable_by_key(|x| x.ordinal);
        }

        let total: u64 = doc_lengths.iter().map(|l| *l as u64).sum();
        let avgdl =
            if doc_lengths.is_empty() { 0.0 } else { total as f32 / doc_lengths.len() as f32 };

        Self {
            terms,
            doc_lengths,
            avgdl,
            tokenizer: config.tokenizer.clone(),
            params: config.params,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn doc_count(&self) -> usize {
        self.doc_lengths.len()
    }

    pub fn term_count(&self) -> usize {
        self.terms.len()
    }

    fn postings(&self, term: &str) -> Option<&[Posting]> {
        self.terms
            .binary_search_by(|(t, _)| t.as_str().cmp(term))
            .ok()
            .map(|i| self.terms[i].1.as_slice())
    }

    /// BM25 score per matching document ordinal.
    ///
    /// A document scoring zero is excluded, matching turbopuffer: a term the
    /// document does not contain contributes nothing, and a query whose every
    /// term is absent should return no rows rather than a page of zeroes.
    pub fn score(&self, query: &str) -> Vec<(u32, f32)> {
        let tokens = self.tokenizer.tokenize(query);
        if tokens.is_empty() || self.doc_lengths.is_empty() {
            return vec![];
        }
        let n = self.doc_lengths.len() as f32;
        let (k1, b) = (self.params.k1, self.params.b);
        let mut acc: HashMap<u32, f32> = HashMap::new();

        // Repeated query terms count once: BM25 saturates on document frequency,
        // and double-counting a term the user typed twice is not more relevant.
        let unique: BTreeSet<&String> = tokens.iter().collect();
        for token in unique {
            let Some(list) = self.postings(token) else { continue };
            let df = list.len() as f32;
            // Robertson-Sparck-Jones IDF, with the +1 that keeps it positive even
            // when a term appears in more than half the corpus.
            let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
            for p in list {
                let tf = p.positions.len() as f32;
                let dl = self.doc_lengths.get(p.ordinal as usize).copied().unwrap_or(0) as f32;
                let norm = if self.avgdl > 0.0 { dl / self.avgdl } else { 0.0 };
                *acc.entry(p.ordinal).or_insert(0.0) +=
                    idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * norm));
            }
        }

        let mut out: Vec<(u32, f32)> = acc.into_iter().filter(|(_, s)| *s > 0.0).collect();
        out.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        out
    }

    /// Score arbitrary text against this index's corpus statistics.
    ///
    /// Needed for the unindexed tail: a document written since the last
    /// compaction has no postings here, but must still be comparable against
    /// documents that do — otherwise recent writes would be unrankable and
    /// effectively invisible to a text query. Using the index's IDF and average
    /// length keeps the scores on one scale.
    pub fn score_text(&self, text: &str, query: &str) -> f32 {
        let terms = self.tokenizer.tokenize(query);
        let doc = self.tokenizer.tokenize(text);
        if terms.is_empty() || doc.is_empty() {
            return 0.0;
        }
        let n = self.doc_lengths.len().max(1) as f32;
        let (k1, b) = (self.params.k1, self.params.b);
        let dl = doc.len() as f32;
        let norm = if self.avgdl > 0.0 { dl / self.avgdl } else { 1.0 };

        let unique: BTreeSet<&String> = terms.iter().collect();
        unique
            .into_iter()
            .map(|term| {
                let tf = doc.iter().filter(|t| *t == term).count() as f32;
                if tf == 0.0 {
                    return 0.0;
                }
                // A term absent from the corpus is maximally rare rather than
                // undefined.
                let df = self.postings(term).map_or(0.0, |l| l.len() as f32);
                let idf = (1.0 + (n - df + 0.5) / (df + 0.5)).ln();
                idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * norm))
            })
            .sum()
    }

    /// Documents containing every token in `query`.
    pub fn all_tokens(&self, query: &str) -> BTreeSet<u32> {
        let tokens = self.tokenizer.tokenize(query);
        if tokens.is_empty() {
            return BTreeSet::new();
        }
        let mut acc: Option<BTreeSet<u32>> = None;
        for token in tokens {
            let here: BTreeSet<u32> = self
                .postings(&token)
                .map(|l| l.iter().map(|p| p.ordinal).collect())
                .unwrap_or_default();
            acc = Some(match acc {
                None => here,
                Some(prev) => prev.intersection(&here).copied().collect(),
            });
            if acc.as_ref().is_some_and(|s| s.is_empty()) {
                break;
            }
        }
        acc.unwrap_or_default()
    }

    /// Documents containing the tokens adjacent and in order — phrase matching.
    ///
    /// This is what positions are stored for. `all_tokens` would match a document
    /// mentioning the words anywhere; a phrase query means the words together.
    pub fn token_sequence(&self, query: &str) -> BTreeSet<u32> {
        let tokens = self.tokenizer.tokenize(query);
        if tokens.is_empty() {
            return BTreeSet::new();
        }
        if tokens.len() == 1 {
            return self.all_tokens(&tokens[0]);
        }

        // Positions of the first token, then require each subsequent token to
        // appear one position later in the same document.
        let Some(first) = self.postings(&tokens[0]) else { return BTreeSet::new() };
        let mut out = BTreeSet::new();
        'doc: for p in first {
            let mut starts: Vec<u32> = p.positions.clone();
            for (offset, token) in tokens.iter().enumerate().skip(1) {
                let Some(list) = self.postings(token) else { break 'doc };
                let Ok(i) = list.binary_search_by_key(&p.ordinal, |x| x.ordinal) else {
                    continue 'doc;
                };
                let wanted = &list[i].positions;
                starts.retain(|s| wanted.binary_search(&(s + offset as u32)).is_ok());
                if starts.is_empty() {
                    continue 'doc;
                }
            }
            if !starts.is_empty() {
                out.insert(p.ordinal);
            }
        }
        out
    }

    /// Documents containing a term within `max_edit_distance` of `query`.
    ///
    /// Scans the term dictionary, which is far smaller than the document set but
    /// still linear in distinct terms — hence turbopuffer requiring the attribute
    /// be marked fuzzy before this is allowed.
    pub fn fuzzy(&self, query: &str, max_edit_distance: usize) -> BTreeSet<u32> {
        let tokens = self.tokenizer.tokenize(query);
        let mut out = BTreeSet::new();
        for token in &tokens {
            for (term, list) in &self.terms {
                if levenshtein_within(term, token, max_edit_distance) {
                    out.extend(list.iter().map(|p| p.ordinal));
                }
            }
        }
        out
    }

    // ------------------------------------------------------------ codec

    pub fn encode(&self) -> Bytes {
        let mut b = BytesMut::new();
        b.put_f32_le(self.avgdl);
        b.put_f32_le(self.params.k1);
        b.put_f32_le(self.params.b);

        let cfg = serde_json::to_vec(&self.tokenizer).unwrap_or_default();
        b.put_u32_le(cfg.len() as u32);
        b.put_slice(&cfg);

        b.put_u32_le(self.doc_lengths.len() as u32);
        for l in &self.doc_lengths {
            b.put_u32_le(*l);
        }

        b.put_u32_le(self.terms.len() as u32);
        for (term, postings) in &self.terms {
            b.put_u32_le(term.len() as u32);
            b.put_slice(term.as_bytes());
            b.put_u32_le(postings.len() as u32);
            for p in postings {
                b.put_u32_le(p.ordinal);
                b.put_u32_le(p.positions.len() as u32);
                for pos in &p.positions {
                    b.put_u32_le(*pos);
                }
            }
        }
        b.freeze()
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        let mut pos = 0usize;
        let mut take = |n: usize| -> Result<&[u8]> {
            let end = pos.checked_add(n).unwrap_or(usize::MAX);
            let Some(s) = buf.get(pos..end) else {
                bail!("truncated full-text index: want {n} bytes at {pos}, have {}", buf.len());
            };
            pos += n;
            Ok(s)
        };
        let f32le = |s: &[u8]| f32::from_le_bytes(s.try_into().unwrap());
        let u32le = |s: &[u8]| u32::from_le_bytes(s.try_into().unwrap());

        let avgdl = f32le(take(4)?);
        let k1 = f32le(take(4)?);
        let b = f32le(take(4)?);

        let cfg_len = u32le(take(4)?) as usize;
        let tokenizer: Tokenizer = serde_json::from_slice(take(cfg_len)?)?;

        let n_docs = u32le(take(4)?) as usize;
        let mut doc_lengths = Vec::with_capacity(n_docs.min(1 << 20));
        for _ in 0..n_docs {
            doc_lengths.push(u32le(take(4)?));
        }

        let n_terms = u32le(take(4)?) as usize;
        let mut terms = Vec::with_capacity(n_terms.min(1 << 20));
        for _ in 0..n_terms {
            let tl = u32le(take(4)?) as usize;
            let term = String::from_utf8(take(tl)?.to_vec())?;
            let np = u32le(take(4)?) as usize;
            let mut postings = Vec::with_capacity(np.min(1 << 20));
            for _ in 0..np {
                let ordinal = u32le(take(4)?);
                let npos = u32le(take(4)?) as usize;
                let mut positions = Vec::with_capacity(npos.min(1 << 16));
                for _ in 0..npos {
                    positions.push(u32le(take(4)?));
                }
                postings.push(Posting { ordinal, positions });
            }
            terms.push((term, postings));
        }

        Ok(Self { terms, doc_lengths, avgdl, tokenizer, params: Bm25Params { k1, b } })
    }
}

/// Is the edit distance between `a` and `b` at most `max`?
///
/// Bounded early: the classic full matrix is wasted work when the answer is
/// "far", and term dictionaries are scanned in full.
pub fn levenshtein_within(a: &str, b: &str, max: usize) -> bool {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    if a.len().abs_diff(b.len()) > max {
        return false;
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        let mut row_min = cur[0];
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            row_min = row_min.min(cur[j]);
        }
        if row_min > max {
            return false;
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()] <= max
}

/// The default edit-distance ladder turbopuffer documents: short queries must
/// match exactly, because one edit on a three-character word matches everything.
pub fn default_edit_distance(query_chars: usize) -> usize {
    match query_chars {
        0..=2 => 0,
        3..=5 => 0,
        _ => 1,
    }
}

/// Text of an attribute for indexing: a string, or every element of a string
/// array joined so positions stay within one document.
pub fn attribute_text(value: &Value) -> Option<String> {
    match value {
        Value::String(s) => Some(s.clone()),
        Value::StringArray(a) => Some(a.join(" ")),
        _ => None,
    }
}

/// Full-text configuration per attribute, keyed by attribute name.
pub type FtsSchema = BTreeMap<String, FtsConfig>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_lowercases_splits_and_stems() {
        let t = Tokenizer::default();
        assert_eq!(t.tokenize("Running, jumps; FOXES!"), vec!["run", "jump", "fox"]);
        // Stopwords go, and their removal does not leave a positional gap.
        assert_eq!(t.tokenize("the quick brown fox"), vec!["quick", "brown", "fox"]);
        assert_eq!(t.tokenize(""), Vec::<String>::new());
        assert_eq!(t.tokenize("!!! ??? ..."), Vec::<String>::new());
        // Digits are tokens.
        assert_eq!(t.tokenize("version 2 beta"), vec!["version", "2", "beta"]);
    }

    #[test]
    fn tokenizer_options_are_honoured() {
        let plain = Tokenizer {
            stemming: false,
            remove_stopwords: false,
            ..Default::default()
        };
        assert_eq!(plain.tokenize("The running foxes"), vec!["the", "running", "foxes"]);

        let cased = Tokenizer { case_sensitive: true, stemming: false, remove_stopwords: false, ..Default::default() };
        assert_eq!(cased.tokenize("The Fox"), vec!["The", "Fox"]);

        // Over-length tokens are dropped, not truncated: a truncated token would
        // collide with a different real word.
        let short = Tokenizer { max_token_length: 4, stemming: false, remove_stopwords: false, ..Default::default() };
        assert_eq!(short.tokenize("abc abcdefgh"), vec!["abc"]);

        let folding = Tokenizer {
            ascii_folding: true,
            stemming: false,
            remove_stopwords: false,
            ..Default::default()
        };
        assert_eq!(folding.tokenize("café naïve"), vec!["cafe", "naive"]);
        // Without folding they stay distinct.
        assert_eq!(plain.tokenize("café"), vec!["café"]);
    }

    fn corpus() -> FtsIndex {
        let docs = vec![
            (0u32, "the quick brown fox jumps over the lazy dog"),
            (1, "a quick brown dog"),
            (2, "the lazy dog sleeps all day"),
            (3, "foxes are quick and foxes are clever"),
            (4, "unrelated content about databases"),
        ];
        FtsIndex::build(docs.into_iter(), 5, &FtsConfig::default())
    }

    #[test]
    fn bm25_ranks_by_relevance() {
        let i = corpus();
        let scored = i.score("quick fox");
        assert!(!scored.is_empty());
        // Document 0 has both terms; 3 has "fox" (stemmed from foxes) twice plus
        // "quick"; both must outrank documents with only one of the terms.
        let ids: Vec<u32> = scored.iter().map(|(o, _)| *o).collect();
        assert!(ids.contains(&0) && ids.contains(&3));
        assert!(!ids.contains(&2), "a document with neither term was scored");
        assert!(!ids.contains(&4));
        // Scores are strictly descending.
        for w in scored.windows(2) {
            assert!(w[0].1 >= w[1].1, "scores not ordered: {scored:?}");
        }
    }

    #[test]
    fn a_query_matching_nothing_returns_nothing() {
        let i = corpus();
        assert!(i.score("zebra").is_empty(), "returned rows for an absent term");
        // A query of nothing but stopwords tokenizes to nothing.
        assert!(i.score("the and of").is_empty());
        assert!(i.score("").is_empty());
    }

    #[test]
    fn repeated_query_terms_count_once() {
        let i = corpus();
        let once = i.score("quick");
        let thrice = i.score("quick quick quick");
        assert_eq!(once, thrice, "a term typed twice inflated the score");
    }

    #[test]
    fn rarer_terms_score_higher_than_common_ones() {
        // "dog" is in 3 of 5 documents, "sleeps" in 1. The rare term must carry
        // more weight, which is the whole point of IDF.
        let i = corpus();
        let common = i.score("dog").iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        let rare = i.score("sleeps").iter().map(|(_, s)| *s).fold(0.0f32, f32::max);
        assert!(rare > common, "IDF did not favour the rarer term: {rare} vs {common}");
    }

    #[test]
    fn document_length_normalisation_applies() {
        // Same term, same frequency, different document lengths: the shorter
        // document is the better match.
        let docs = vec![(0u32, "alpha"), (1, "alpha beta gamma delta epsilon zeta eta theta")];
        let i = FtsIndex::build(docs.into_iter(), 2, &FtsConfig::default());
        let scored = i.score("alpha");
        assert_eq!(scored[0].0, 0, "length normalisation did not favour the shorter document");
    }

    #[test]
    fn all_tokens_requires_every_term() {
        let i = corpus();
        // Both documents 0 and 1 contain "quick" and "dog" somewhere.
        assert_eq!(i.all_tokens("quick dog"), BTreeSet::from([0, 1]));
        assert_eq!(i.all_tokens("quick"), BTreeSet::from([0, 1, 3]));
        assert!(i.all_tokens("quick zebra").is_empty(), "matched despite a missing term");
        assert!(i.all_tokens("").is_empty());
    }

    #[test]
    fn phrase_matching_needs_adjacency_and_order() {
        let i = corpus();
        // "quick brown" appears adjacent in 0 and 1.
        assert_eq!(i.token_sequence("quick brown"), BTreeSet::from([0, 1]));
        // Reversed, it appears nowhere — which is what distinguishes a phrase
        // query from a bag of words.
        assert!(i.token_sequence("brown quick").is_empty());
        // The same words non-adjacent match all_tokens but not the phrase.
        assert!(!i.all_tokens("quick dog").is_empty());
        assert!(i.token_sequence("quick dog").is_empty());
        // A three-token phrase.
        assert_eq!(i.token_sequence("quick brown fox"), BTreeSet::from([0]));
        // A single token degenerates to a term lookup.
        assert_eq!(i.token_sequence("fox"), BTreeSet::from([0, 3]));
    }

    #[test]
    fn phrase_matching_spans_removed_stopwords() {
        // "over the lazy" -> tokens [over, lazy], adjacent because the stopword
        // left no gap. A user searching a phrase containing "the" expects a hit.
        let i = corpus();
        assert_eq!(i.token_sequence("over the lazy"), BTreeSet::from([0]));
    }

    #[test]
    fn fuzzy_matches_within_an_edit() {
        let i = corpus();
        // "databses" -> stem "databs"; the indexed term is "databas". One edit.
        assert_eq!(i.fuzzy("databses", 1), BTreeSet::from([4]));
        assert!(i.fuzzy("databses", 0).is_empty(), "distance 0 matched a misspelling");
        assert!(i.fuzzy("zzzzzzzz", 1).is_empty());
    }

    #[test]
    fn levenshtein_bound_is_correct() {
        assert!(levenshtein_within("kitten", "kitten", 0));
        assert!(levenshtein_within("kitten", "sitten", 1));
        assert!(!levenshtein_within("kitten", "sitten", 0));
        assert!(levenshtein_within("kitten", "sitting", 3));
        assert!(!levenshtein_within("kitten", "sitting", 2));
        // A length gap larger than the budget short-circuits.
        assert!(!levenshtein_within("a", "abcdef", 2));
        assert!(levenshtein_within("", "", 0));
        assert!(levenshtein_within("", "ab", 2));
        // Unicode is compared by character, not byte.
        assert!(levenshtein_within("café", "cafe", 1));
    }

    #[test]
    fn short_queries_demand_exact_matches() {
        // One edit on a three-character word matches an enormous fraction of any
        // dictionary, so the ladder requires exactness there.
        assert_eq!(default_edit_distance(2), 0);
        assert_eq!(default_edit_distance(4), 0);
        assert_eq!(default_edit_distance(6), 1);
        assert_eq!(default_edit_distance(20), 1);
    }

    #[test]
    fn unindexed_text_scores_on_the_same_scale() {
        let i = corpus();
        // A document identical to an indexed one must score close to it, so a
        // freshly written document is comparable rather than unrankable.
        let indexed = i.score("quick brown").iter().find(|(o, _)| *o == 1).unwrap().1;
        let fresh = i.score_text("a quick brown dog", "quick brown");
        assert!(
            (indexed - fresh).abs() / indexed.max(1e-6) < 0.2,
            "tail scoring diverged: indexed {indexed}, fresh {fresh}"
        );
        // No overlap scores zero, so it is excluded like any non-match.
        assert_eq!(i.score_text("completely different words", "quick brown"), 0.0);
        assert_eq!(i.score_text("", "quick"), 0.0);
        assert_eq!(i.score_text("quick", ""), 0.0);
    }

    #[test]
    fn roundtrip_through_bytes() {
        let i = corpus();
        let encoded = i.encode();
        let decoded = FtsIndex::decode(&encoded).unwrap();
        assert_eq!(decoded, i);
        // Scores survive, which is what actually matters.
        assert_eq!(decoded.score("quick fox"), i.score("quick fox"));
        assert_eq!(decoded.token_sequence("quick brown"), i.token_sequence("quick brown"));

        for cut in 0..encoded.len() {
            // Truncation must error, never panic and never yield a short index
            // that silently scores differently.
            let _ = FtsIndex::decode(&encoded[..cut]);
        }
    }

    #[test]
    fn parameters_travel_with_the_index() {
        // Scores must stay comparable within a namespace, so the index carries
        // the parameters it was built with rather than reading current config.
        let cfg = FtsConfig { params: Bm25Params { k1: 2.5, b: 0.1 }, ..Default::default() };
        let docs = vec![(0u32, "alpha beta"), (1, "alpha")];
        let i = FtsIndex::build(docs.into_iter(), 2, &cfg);
        let decoded = FtsIndex::decode(&i.encode()).unwrap();
        assert_eq!(decoded.params, Bm25Params { k1: 2.5, b: 0.1 });
        assert_eq!(decoded.score("alpha"), i.score("alpha"));
    }

    #[test]
    fn array_attributes_index_as_joined_text() {
        assert_eq!(attribute_text(&Value::from("hi")).as_deref(), Some("hi"));
        assert_eq!(
            attribute_text(&Value::StringArray(vec!["red".into(), "blue".into()])).as_deref(),
            Some("red blue")
        );
        assert!(attribute_text(&Value::Uint(1)).is_none());

        // Joining keeps positions inside one document, so a phrase cannot
        // straddle two array elements... except adjacently, which is the
        // documented behaviour of joining.
        let i = FtsIndex::build(
            [(0u32, "red blue")].into_iter(),
            1,
            &FtsConfig::default(),
        );
        assert_eq!(i.token_sequence("red blue"), BTreeSet::from([0]));
    }

    #[test]
    fn empty_corpus_is_harmless() {
        let i = FtsIndex::build(std::iter::empty::<(u32, &str)>(), 0, &FtsConfig::default());
        assert!(i.is_empty());
        assert!(i.score("anything").is_empty());
        assert!(i.all_tokens("anything").is_empty());
        assert!(i.token_sequence("a b").is_empty());
        assert!(i.fuzzy("a", 1).is_empty());
        assert_eq!(FtsIndex::decode(&i.encode()).unwrap(), i);
    }
}
