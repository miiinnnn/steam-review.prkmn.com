import { getModelInfo } from '../web/lib.mjs';

export default function handler(req, res) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method Not Allowed' });
  }
  res.setHeader('Cache-Control', 'public, max-age=3600');
  return res.status(200).json(getModelInfo());
}
