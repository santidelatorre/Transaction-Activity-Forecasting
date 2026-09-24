/** Translate the former Laravel URLs into small client-side hash routes. */
export function routeHref(href) {
  const parsed = new URL(href, window.location.origin);
  let route = parsed.pathname.replace(/^\/insights/, '') || '/overview';
  if (route === '/client') route = '/client';
  return `#${route}${parsed.search}`;
}

export function Link({ href, ...props }) {
  return <a href={routeHref(href)} {...props} />;
}
