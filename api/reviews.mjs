import { analyzeGameReviews } from '../web/lib.mjs';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }
  try {
    const rawAppId = String(req.query?.appid || '').trim();
    const gameName = String(req.query?.name || '').trim();
    const limit = Math.min(1000, Math.max(20, Number(req.query?.limit || 1000)));
    const cursor = String(req.query?.cursor || '*').trim() || '*';

    if (!/^\d{1,10}$/.test(rawAppId)) {
      return res.status(400).json({ error: '올바른 Steam 게임 ID가 아닙니다.' });
    }
    const { status, body } = await analyzeGameReviews({
      appid: Number(rawAppId),
      gameName,
      limit,
      cursor,
    });
    res.setHeader('Cache-Control', 'no-store');
    return res.status(status).json(body);
  } catch (err) {
    const msg =
      err?.name === 'TimeoutError'
        ? 'Steam 서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.'
        : err?.message || '서버 오류가 발생했습니다.';
    return res.status(502).json({ error: msg });
  }
}
