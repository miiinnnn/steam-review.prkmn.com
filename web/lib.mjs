import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const MODEL_PATH = path.join(ROOT, 'model', 'bug_opt_classifier.json');
const METRICS_PATH = path.join(ROOT, 'model', 'metrics.json');

export const model = JSON.parse(fs.readFileSync(MODEL_PATH, 'utf8'));
export const metrics = fs.existsSync(METRICS_PATH)
  ? JSON.parse(fs.readFileSync(METRICS_PATH, 'utf8'))
  : null;

const vocabIndex = new Map(model.vocab.map((token, idx) => [token, idx]));
const [minN, maxN] = model.ngram_range || [2, 4];
const CLAUSE_SPLIT_RE = /[\n.!?。]+|\s+(?:근데|그런데|하지만|다만|그러나|그래도)\s+/u;
const tagMatchers = (model.issue_tags || []).map(({ tag, pattern }) => ({
  tag,
  re: new RegExp(pattern, 'iu'),
}));

export function scoreClause(clause) {
  const norm = clause.toLowerCase().replace(/\s+/gu, ' ').trim();
  if (!norm) return 0;
  const chars = Array.from(norm);
  const tf = new Map();
  for (let n = minN; n <= maxN; n++) {
    for (let i = 0; i + n <= chars.length; i++) {
      const gram = chars.slice(i, i + n).join('');
      const idx = vocabIndex.get(gram);
      if (idx !== undefined) tf.set(idx, (tf.get(idx) || 0) + 1);
    }
  }
  let sqSum = 0;
  for (const [idx, count] of tf) {
    const val = (1 + Math.log(count)) * model.idf[idx];
    tf.set(idx, val);
    sqSum += val * val;
  }
  const normFactor = Math.sqrt(sqSum) || 1;
  let logit = model.bias;
  for (const [idx, val] of tf) {
    logit += (model.weights[idx] * val) / normFactor;
  }
  return 1 / (1 + Math.exp(-logit));
}

export function classifyReview(reviewText) {
  const parts = reviewText
    .split(CLAUSE_SPLIT_RE)
    .map((s) => s.trim())
    .filter(Boolean);
  const clauses = parts.length ? parts : [reviewText.trim()];

  let maxProb = 0;
  let bestClause = clauses[0] || '';
  for (const clause of clauses) {
    const prob = scoreClause(clause);
    if (prob > maxProb) {
      maxProb = prob;
      bestClause = clause;
    }
  }

  const isBugOpt = maxProb >= model.threshold;
  const tags = [];
  if (isBugOpt) {
    for (const { tag, re } of tagMatchers) {
      if (re.test(bestClause) || re.test(reviewText)) {
        tags.push(tag);
      }
    }
    if (!tags.length) tags.push('기타 기술 이슈');
  }

  return {
    isBugOpt,
    probability: Number(maxProb.toFixed(4)),
    matchedClause: bestClause,
    tags,
  };
}

const KO_ALIAS_MAP = [
  ['엘든링', 'ELDEN RING'],
  ['스타듀밸리', 'Stardew Valley'],
  ['배틀그라운드', 'PUBG'],
  ['배그', 'PUBG'],
  ['사이버펑크', 'Cyberpunk'],
  ['사펑', 'Cyberpunk'],
  ['몬스터헌터', 'Monster Hunter'],
  ['몬헌', 'Monster Hunter'],
  ['발더스게이트', "Baldur's Gate"],
  ['위쳐', 'Witcher'],
  ['다크소울', 'Dark Souls'],
  ['세키로', 'Sekiro'],
  ['호그와트', 'Hogwarts'],
  ['팰월드', 'Palworld'],
  ['팔월드', 'Palworld'],
  ['레드데드리뎀션', 'Red Dead Redemption'],
  ['레데리', 'Red Dead Redemption'],
  ['그랜드테프트오토', 'Grand Theft Auto'],
  ['그타', 'Grand Theft Auto'],
  ['헬다이버즈', 'Helldivers'],
  ['리썰컴퍼니', 'Lethal Company'],
  ['테라리아', 'Terraria'],
  ['할로우나이트', 'Hollow Knight'],
  ['할나', 'Hollow Knight'],
  ['슬레이더스파이어', 'Slay the Spire'],
  ['하데스', 'Hades'],
  ['포탈', 'Portal'],
  ['하프라이프', 'Half-Life'],
  ['문명', 'Civilization'],
  ['에이펙스', 'Apex Legends'],
  ['카운터스트라이크', 'Counter-Strike'],
  ['카스', 'Counter-Strike'],
  ['도타', 'Dota'],
  ['데드바이데이라이트', 'Dead by Daylight'],
  ['데바데', 'Dead by Daylight'],
  ['발헤임', 'Valheim'],
  ['러스트', 'Rust'],
  ['아크', 'ARK'],
  ['림월드', 'RimWorld'],
  ['팩토리오', 'Factorio'],
  ['노맨즈스카이', "No Man's Sky"],
  ['페르소나', 'Persona'],
  ['파이널판타지', 'Final Fantasy'],
  ['파판', 'Final Fantasy'],
  ['바이오하자드', 'Resident Evil'],
  ['바하', 'Resident Evil'],
  ['데빌메이크라이', 'Devil May Cry'],
  ['스파이더맨', 'Spider-Man'],
  ['갓오브워', 'God of War'],
  ['라스트오브어스', 'The Last of Us'],
  ['라오어', 'The Last of Us'],
  ['호라이즌', 'Horizon'],
  ['고스트오브쓰시마', 'Ghost of Tsushima'],
  ['아머드코어', 'Armored Core'],
  ['검은신화오공', 'Black Myth'],
  ['킹덤컴', 'Kingdom Come'],
  ['스카이림', 'Skyrim'],
  ['폴아웃', 'Fallout'],
  ['스타필드', 'Starfield'],
  ['어쌔신크리드', "Assassin's Creed"],
  ['유로트럭', 'Euro Truck'],
  ['심즈', 'The Sims'],
  ['풋볼매니저', 'Football Manager'],
  ['철권', 'TEKKEN'],
  ['스트리트파이터', 'Street Fighter'],
  ['길티기어', 'Guilty Gear'],
  ['프로젝트좀보이드', 'Project Zomboid'],
  ['좀보이드', 'Project Zomboid'],
  ['돈스타브', "Don't Starve"],
  ['언더테일', 'Undertale'],
  ['오버워치', 'Overwatch'],
  ['이터널리턴', 'Eternal Return'],
  ['디제이맥스', 'DJMAX'],
  ['데이브더다이버', 'DAVE THE DIVER'],
  ['산나비', 'SANABI'],
  ['스텔라블레이드', 'Stellar Blade'],
  ['피의거짓', 'Lies of P'],
  ['p의거짓', 'Lies of P'],
  ['로스트아크', 'Lost Ark'],
];

export async function searchSteamGames(query) {
  const q = query.trim();
  if (!q) return [];

  const compact = q.toLowerCase().replace(/\s+/gu, '');
  const terms = [q];
  if (/[가-힣]/u.test(compact)) {
    for (const [ko, en] of KO_ALIAS_MAP) {
      if (ko.includes(compact) || compact.includes(ko)) {
        if (!terms.includes(en)) terms.push(en);
        if (terms.length >= 4) break;
      }
    }
  }

  const results = [];
  const seen = new Set();

  async function fetchTerm(term, lang = 'korean', cc = 'KR') {
    const url = `https://store.steampowered.com/api/storesearch/?term=${encodeURIComponent(term)}&l=${lang}&cc=${cc}`;
    const res = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' },
      signal: AbortSignal.timeout(10000),
    });
    if (!res.ok) throw new Error(`Steam 검색 API 오류 (${res.status})`);
    const data = await res.json();
    for (const item of data.items || []) {
      if (!item.id || seen.has(item.id)) continue;
      seen.add(item.id);
      results.push({
        appid: item.id,
        name: item.name,
        tinyImage: item.tiny_image || '',
      });
    }
  }

  for (const term of terms) {
    try {
      await fetchTerm(term, 'korean', 'KR');
    } catch {
      // continue to next alias
    }
    if (results.length >= 10) break;
  }

  if (results.length < 5) {
    try {
      await fetchTerm(terms[terms.length - 1], 'english', 'US');
    } catch {
      // ignore fallback errors
    }
  }
  return results.slice(0, 10);
}

export async function fetchSteamReviews(appid, maxReviews = 1000, startCursor = '*') {
  const reviews = [];
  const seen = new Set();
  const visitedCursors = new Set();
  let cursor = startCursor || '*';
  let totalReviews = null;
  let hasMore = false;
  const maxPages = Math.ceil(maxReviews / 100) + 2;

  for (let page = 0; page < maxPages && reviews.length < maxReviews && cursor; page++) {
    if (visitedCursors.has(cursor)) {
      hasMore = false;
      break;
    }
    visitedCursors.add(cursor);

    const payload = {
      appid,
      languages: ['koreana'],
      filter: 1,
      purchase_type: 1,
      review_type: 0,
      num_per_page: Math.min(100, maxReviews - reviews.length),
      cursor,
    };
    const url =
      'https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/?input_json=' +
      encodeURIComponent(JSON.stringify(payload));

    let res;
    for (let attempt = 0; attempt < 3; attempt++) {
      res = await fetch(url, {
        headers: { 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)' },
        signal: AbortSignal.timeout(15000),
      });
      if (res.status !== 429) break;
      await new Promise((resolve) => setTimeout(resolve, 1000 * (attempt + 1)));
    }

    if (!res.ok) {
      if (reviews.length > 0) {
        hasMore = true;
        break;
      }
      throw new Error(
        res.status === 429
          ? 'Steam 요청이 일시적으로 많습니다. 잠시 후 다시 시도해 주세요.'
          : `Steam 리뷰 API 오류 (${res.status})`
      );
    }
    const body = (await res.json())?.response;
    if (totalReviews === null && body?.query_summary?.total_reviews != null) {
      totalReviews = Number(body.query_summary.total_reviews);
    }
    const batch = body?.reviews || [];
    if (!batch.length) {
      hasMore = false;
      cursor = '';
      break;
    }

    for (const r of batch) {
      if (!r.recommendationid || seen.has(r.recommendationid)) continue;
      if (r.language && r.language !== 'koreana') continue;
      const text = (r.review || '').trim();
      if (!text) continue;
      seen.add(r.recommendationid);
      reviews.push({
        id: String(r.recommendationid),
        text,
        recommended: Boolean(r.voted_up),
        playtimeHours: Number(((r.author?.playtime_at_review || r.author?.playtime_forever || 0) / 60).toFixed(1)),
        createdAt: r.timestamp_created || 0,
        helpfulVotes: Number(r.votes_up || 0),
      });
    }

    const nextCursor = body?.cursor || '';
    if (!nextCursor || visitedCursors.has(nextCursor)) {
      cursor = '';
      hasMore = false;
      break;
    }
    cursor = nextCursor;
    hasMore = true;

    if (reviews.length < maxReviews) {
      await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }

  return { reviews, nextCursor: cursor, hasMore: Boolean(hasMore && cursor), totalReviews };
}

export async function analyzeGameReviews({ appid, gameName, limit = 1000, cursor = '*' }) {
  const { reviews: allReviews, nextCursor, hasMore, totalReviews } = await fetchSteamReviews(
    appid,
    limit,
    cursor
  );

  if (!allReviews.length && cursor === '*') {
    return { status: 404, body: { error: '이 게임의 한국어 리뷰를 찾지 못했습니다.' } };
  }

  const bugOptReviews = [];
  for (const r of allReviews) {
    const pred = classifyReview(r.text);
    if (pred.isBugOpt) {
      bugOptReviews.push({
        ...r,
        probability: pred.probability,
        matchedClause: pred.matchedClause,
        tags: pred.tags,
      });
    }
  }

  bugOptReviews.sort((a, b) => b.probability - a.probability);

  return {
    status: 200,
    body: {
      appid,
      gameName: gameName || `Steam App ${appid}`,
      totalScanned: allReviews.length,
      totalReviews,
      issueCount: bugOptReviews.length,
      issueRatio: allReviews.length
        ? Number(((bugOptReviews.length / allReviews.length) * 100).toFixed(1))
        : 0,
      nextCursor,
      hasMore,
      threshold: model.threshold,
      reviews: bugOptReviews,
    },
  };
}

export function getModelInfo() {
  return {
    trainedReviews: model.trained_reviews,
    threshold: model.threshold,
    labels: model.labels,
    metrics: metrics ? metrics.test : null,
  };
}
