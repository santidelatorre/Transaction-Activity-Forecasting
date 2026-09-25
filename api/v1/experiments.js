const dashboard = require('../data/dashboard.json');

function integer(value, fallback, min, max) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < min || parsed > max) return fallback;
  return parsed;
}

module.exports = (req, res) => {
  const query = req.query || {};
  const limit = integer(query.limit, 100, 1, 100);
  const offset = integer(query.offset, 0, 0, 1_000_000);
  const rows = Array.isArray(dashboard.experiments) ? dashboard.experiments : [];
  res.setHeader('Cache-Control', 'public, max-age=60');
  res.status(200).json({
    available: rows.length > 0 || Boolean(dashboard.v4_experiments?.length),
    experiments: rows.slice(offset, offset + limit),
    limit,
    offset,
    comparison_group: dashboard.comparison_group,
  });
};
