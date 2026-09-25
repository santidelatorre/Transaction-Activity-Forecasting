const dashboard = require('../data/dashboard.json');

module.exports = (_req, res) => {
  res.setHeader('Cache-Control', 'public, max-age=300');
  res.status(200).json(dashboard);
};
