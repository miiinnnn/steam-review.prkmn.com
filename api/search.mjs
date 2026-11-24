import { searchSteamGames } from '../web/lib.mjs';

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }
  try {
    const q = String(req.query?.q || '').trim();
    if (!q) return res.status(200).json({ items: [] });
    const items = await searchSteamGames(q);
    res.setHeader('Cache-Control', 'no-store');
    return res.status(200).json({ items });
  } catch (err) {
    const msg =
      err?.name === 'TimeoutError'
        ? 'Steam 검색 응답 시간이 초과되었습니다.'
        : err?.message || '검색 중 오류가 발생했습니다.';
    return res.status(502).json({ error: msg });
  }
}
