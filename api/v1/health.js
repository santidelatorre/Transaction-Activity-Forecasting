const health = require('../data/health.json');

module.exports = (_req, res) => {
  res.setHeader('Cache-Control', 'public, max-age=60');
  res.status(200).json(health);
};
