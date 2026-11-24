import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { analyzeGameReviews, getModelInfo, searchSteamGames } from './lib.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const HTML_PATH = path.resolve(__dirname, '..', 'public', 'index.html');

function sendJson(res, status, payload) {
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  res.end(JSON.stringify(payload));
}

const server = http.createServer(async (req, res) => {
  try {
    const reqUrl = new URL(req.url || '/', `http://${req.headers.host || 'localhost'}`);

    if (req.method !== 'GET') {
      return sendJson(res, 405, { error: 'Method Not Allowed' });
    }

    if (reqUrl.pathname === '/' || reqUrl.pathname === '/index.html') {
      const indexHtml = fs.readFileSync(HTML_PATH, 'utf8');
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      return res.end(indexHtml);
    }

    if (reqUrl.pathname === '/api/model-info') {
      return sendJson(res, 200, getModelInfo());
    }

    if (reqUrl.pathname === '/api/search') {
      const q = (reqUrl.searchParams.get('q') || '').trim();
      if (!q) return sendJson(res, 200, { items: [] });
      const items = await searchSteamGames(q);
      return sendJson(res, 200, { items });
    }

    if (reqUrl.pathname === '/api/reviews') {
      const rawAppId = (reqUrl.searchParams.get('appid') || '').trim();
      const gameName = (reqUrl.searchParams.get('name') || '').trim();
      const limit = Math.min(1000, Math.max(20, Number(reqUrl.searchParams.get('limit') || 1000)));
      const cursor = (reqUrl.searchParams.get('cursor') || '*').trim() || '*';

      if (!/^\d{1,10}$/.test(rawAppId)) {
        return sendJson(res, 400, { error: '올바른 Steam 게임 ID가 아닙니다.' });
      }
      const { status, body } = await analyzeGameReviews({
        appid: Number(rawAppId),
        gameName,
        limit,
        cursor,
      });
      return sendJson(res, status, body);
    }

    return sendJson(res, 404, { error: 'Not Found' });
  } catch (err) {
    const msg =
      err?.name === 'TimeoutError'
        ? 'Steam 서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해 주세요.'
        : err?.message || '서버 오류가 발생했습니다.';
    return sendJson(res, 502, { error: msg });
  }
});

const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || '127.0.0.1';
server.listen(PORT, HOST, () => {
  console.log(`Steam Bug/Optimization Review Analyzer running at http://${HOST}:${PORT}`);
});
