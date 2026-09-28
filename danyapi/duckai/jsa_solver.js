"use strict";

// Evaluates the Duck.ai attestation script (served base64 in the
// x-vqd-hash-1 response header) against a minimal Chrome-on-Windows DOM shim
// and prints the raw attestation object as JSON on stdout.
//
// The script is freshly obfuscated on every response: the string table, its
// rotation, the checksum bases and even the set of probes differ per request.
// Evaluating the real script is the only way to stay correct across rotations.
// Node has no DOM, so the shim below reproduces the browser surface the
// attestation touches: the prototype hierarchy the probes instanceof against,
// the error/stack API, the innerHTML fragment parser, layout stubs and the
// frame-global leak probe.

const fs = require("fs");
const vm = require("vm");

const VOID_TAGS = new Set([
  "area", "base", "br", "col", "embed", "hr", "img", "input",
  "link", "meta", "param", "source", "track", "wbr",
]);

const IMPLIED_END_TAGS = new Set([
  "dd", "dt", "li", "optgroup", "option", "p", "rb", "rp", "rt", "rtc",
]);

const CLOSES_P = new Set([
  "address", "article", "aside", "blockquote", "center", "details", "dialog",
  "dir", "div", "dl", "fieldset", "figcaption", "figure", "footer", "form",
  "h1", "h2", "h3", "h4", "h5", "h6", "header", "hgroup", "hr", "listing",
  "main", "menu", "nav", "ol", "p", "plaintext", "pre", "search", "section",
  "summary", "table", "ul", "xmp",
]);

const LIST_ITEM_TAGS = ["li", "dd", "dt"];

const MATH_TEXT_INTEGRATION = new Set(["mi", "mo", "mn", "ms", "mtext"]);

const FOREIGN_ROOTS = new Set(["math", "svg"]);

const RAW_TEXT_TAGS = new Set(["script", "style"]);

const RCDATA_TAGS = new Set(["textarea", "title"]);

const HTML_TAGS = new Set([
  "a", "abbr", "address", "area", "article", "aside", "audio", "b", "base",
  "bdi", "bdo", "blockquote", "body", "br", "button", "canvas", "caption",
  "cite", "code", "col", "colgroup", "data", "datalist", "dd", "del",
  "details", "dfn", "dialog", "div", "dl", "dt", "em", "embed", "fieldset",
  "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5",
  "h6", "head", "header", "hgroup", "hr", "html", "i", "iframe", "img",
  "input", "ins", "kbd", "label", "legend", "li", "link", "main", "map",
  "mark", "menu", "meta", "meter", "nav", "noscript", "object", "ol",
  "optgroup", "option", "output", "p", "param", "picture", "pre", "progress",
  "q", "rp", "rt", "ruby", "s", "samp", "script", "search", "section",
  "select", "slot", "small", "source", "span", "strong", "style", "sub",
  "summary", "sup", "table", "tbody", "td", "template", "textarea", "tfoot",
  "th", "thead", "time", "title", "tr", "track", "u", "ul", "var", "video",
  "wbr",
]);

const ENTITIES = {
  amp: "&", lt: "<", gt: ">", quot: '"', apos: "'", nbsp: " ",
  copy: "©", reg: "®", hellip: "…", mdash: "—", ndash: "–",
  laquo: "«", raquo: "»", trade: "™", deg: "°", middot: "·",
};

function decodeEntities(text) {
  return text.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);/g, (match, body) => {
    if (body[0] === "#") {
      const hex = body[1] === "x" || body[1] === "X";
      const code = Number.parseInt(hex ? body.slice(2) : body.slice(1), hex ? 16 : 10);
      return Number.isFinite(code) && code > 0 && code <= 0x10ffff ? String.fromCodePoint(code) : match;
    }
    const named = ENTITIES[body.toLowerCase()];
    return named === undefined ? match : named;
  });
}

function makeStorage() {
  const map = new Map();
  return {
    get length() { return map.size; },
    key(index) { return Array.from(map.keys())[index] ?? null; },
    getItem(key) { const k = String(key); return map.has(k) ? map.get(k) : null; },
    setItem(key, value) { map.set(String(key), String(value)); },
    removeItem(key) { map.delete(String(key)); },
    clear() { map.clear(); },
  };
}

function parseStyle(cssText) {
  const declarations = new Map();
  for (const part of String(cssText || "").split(";")) {
    const at = part.indexOf(":");
    if (at === -1) continue;
    const name = part.slice(0, at).trim().toLowerCase();
    const value = part.slice(at + 1).trim();
    if (name) declarations.set(name, value);
  }
  return declarations;
}

const PX_PER_EM = 16;
const CHAR_WIDTH_RATIO = 0.5;
const LINE_HEIGHT_RATIO = 1.2;

function lengthOf(value, fallback) {
  if (typeof value !== "string") return fallback;
  const text = value.trim();
  if (text === "") return fallback;
  if (text.endsWith("px")) {
    const number = Number.parseFloat(text.slice(0, -2));
    return Number.isFinite(number) ? number : fallback;
  }
  if (text.endsWith("rem")) {
    const number = Number.parseFloat(text.slice(0, -3));
    return Number.isFinite(number) ? number * PX_PER_EM : fallback;
  }
  if (text.endsWith("em")) {
    const number = Number.parseFloat(text.slice(0, -2));
    return Number.isFinite(number) ? number * PX_PER_EM : fallback;
  }
  const plain = Number.parseFloat(text);
  return Number.isFinite(plain) ? plain : fallback;
}

// Minimal box model: just enough for the layout probes the attestation runs
// (a styled, text-bearing element must report a non-zero box, a display:none
// element must report zero, and computed style must echo the declarations).
function layoutOf(element) {
  const declared = parseStyle(element.style && element.style.cssText);
  const inline = parseStyle(element.style && element.style.getPropertyValue ? element.style.getPropertyValue("") : "");
  for (const [name, value] of inline) declared.set(name, value);
  for (const [name, value] of Object.entries(element.style && element.style.declarations ? element.style.declarations : {})) {
    declared.set(name, value);
  }
  const display = declared.get("display") || "block";
  if (display === "none") {
    return { display, width: 0, height: 0, scrollWidth: 0, scrollHeight: 0, rect: { x: 0, y: 0, width: 0, height: 0, top: 0, left: 0, right: 0, bottom: 0 } };
  }
  const padding = lengthOf(declared.get("padding"), 0);
  const fontSize = lengthOf(declared.get("font-size"), PX_PER_EM);
  const text = element.textContent || "";
  const lines = text ? text.split("\n") : [];
  const lineHeight = fontSize * LINE_HEIGHT_RATIO;
  let contentWidth = 0;
  for (const line of lines) {
    contentWidth = Math.max(contentWidth, line.length * fontSize * CHAR_WIDTH_RATIO);
  }
  const explicitWidth = declared.has("width") ? lengthOf(declared.get("width"), 0) : null;
  const explicitHeight = declared.has("height") ? lengthOf(declared.get("height"), 0) : null;
  const contentHeight = lines.length ? lines.length * lineHeight : 0;
  const width = Math.round((explicitWidth === null ? contentWidth : explicitWidth) + padding * 2);
  const height = Math.round((explicitHeight === null ? contentHeight : explicitHeight) + padding * 2);
  return {
    display,
    width,
    height,
    scrollWidth: width,
    scrollHeight: Math.max(height, Math.round(contentHeight + padding * 2)),
    rect: { x: 0, y: 0, width, height, top: 0, left: 0, right: width, bottom: height },
  };
}

function createStyle() {
  const declarations = new Map();
  const style = {
    setProperty(name, value) { declarations.set(String(name), String(value)); },
    getPropertyValue(name) {
      if (name === "" || name === null || name === undefined) {
        return Array.from(declarations, ([name, value]) => `${name}: ${value}`).join("; ");
      }
      return declarations.get(String(name)) ?? "";
    },
    removeProperty(name) { declarations.delete(String(name)); },
    get cssText() { return style.getPropertyValue(""); },
    set cssText(value) {
      declarations.clear();
      for (const [name, entry] of parseStyle(value)) declarations.set(name, entry);
    },
    get length() { return declarations.size; },
    declarations,
  };
  return style;
}

class DomNode {
  constructor(doc, nodeType, nodeName) {
    this.ownerDocument = doc;
    this.parentNode = null;
    this.childNodes = [];
    this.nodeType = nodeType;
    this.nodeName = nodeName;
  }

  get children() {
    return this.childNodes.filter((node) => node.nodeType === 1);
  }

  get firstChild() { return this.childNodes[0] ?? null; }

  get lastChild() { return this.childNodes[this.childNodes.length - 1] ?? null; }

  get firstElementChild() { return this.children[0] ?? null; }

  get lastElementChild() { const list = this.children; return list[list.length - 1] ?? null; }

  get childElementCount() { return this.children.length; }

  get nextSibling() {
    if (this.parentNode === null) return null;
    const siblings = this.parentNode.childNodes;
    return siblings[siblings.indexOf(this) + 1] ?? null;
  }

  get previousSibling() {
    if (this.parentNode === null) return null;
    const siblings = this.parentNode.childNodes;
    return siblings[siblings.indexOf(this) - 1] ?? null;
  }

  get nextElementSibling() { return this.nextSibling && this.nextSibling.nodeType === 1 ? this.nextSibling : null; }

  get previousElementSibling() { return this.previousSibling && this.previousSibling.nodeType === 1 ? this.previousSibling : null; }

  get rootNode() { let node = this; while (node.parentNode) node = node.parentNode; return node; }

  appendChild(child) {
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }

  insertBefore(child, reference) {
    if (reference === null || reference === undefined) return this.appendChild(child);
    const index = this.childNodes.indexOf(reference);
    if (index === -1) return this.appendChild(child);
    if (child.parentNode) child.parentNode.removeChild(child);
    child.parentNode = this;
    this.childNodes.splice(index, 0, child);
    return child;
  }

  removeChild(child) {
    const index = this.childNodes.indexOf(child);
    if (index !== -1) {
      this.childNodes.splice(index, 1);
      child.parentNode = null;
    }
    return child;
  }

  replaceChild(next, previous) {
    const index = this.childNodes.indexOf(previous);
    if (index === -1) return previous;
    if (next.parentNode) next.parentNode.removeChild(next);
    next.parentNode = this;
    this.childNodes.splice(index, 1, next);
    previous.parentNode = null;
    return previous;
  }

  contains(other) {
    let node = other;
    while (node) {
      if (node === this) return true;
      node = node.parentNode;
    }
    return false;
  }

  hasChildNodes() { return this.childNodes.length > 0; }

  cloneNode(deep) {
    const copy = this.constructor === TextNode
      ? new TextNode(this.ownerDocument, this.data)
      : this.ownerDocument.createElement(this.localName ?? "div");
    for (const [name, value] of Object.entries(this.attrs ?? {})) copy.attrs[name] = value;
    if (deep) for (const child of this.childNodes) copy.appendChild(child.cloneNode(true));
    return copy;
  }

  get textContent() { return this.childNodes.map((node) => node.textContent).join(""); }

  set textContent(value) {
    this.childNodes = [];
    if (value !== "" && value !== null && value !== undefined) {
      this.appendChild(this.ownerDocument.createTextNode(String(value)));
    }
  }

  get innerHTML() { return this.childNodes.map((node) => node.outerHTML).join(""); }

  set innerHTML(html) {
    this.childNodes = [];
    for (const child of parseFragment(this.ownerDocument, String(html))) {
      child.parentNode = this;
      this.childNodes.push(child);
    }
  }

  get innerText() { return this.textContent; }

  get outerHTML() { return this.innerHTML; }

  get offsetWidth() { return layoutOf(this).width; }

  get offsetHeight() { return layoutOf(this).height; }

  get offsetTop() { return 0; }

  get offsetLeft() { return 0; }

  get clientWidth() { return layoutOf(this).width; }

  get clientHeight() { return layoutOf(this).height; }

  get scrollWidth() { return layoutOf(this).scrollWidth; }

  get scrollHeight() { return layoutOf(this).scrollHeight; }

  get scrollTop() { return 0; }

  get scrollLeft() { return 0; }

  getBoundingClientRect() {
    return { ...layoutOf(this).rect, toJSON() { return {}; } };
  }

  getClientRects() { return { length: 0, item: () => null }; }

  scrollIntoView() {}

  get querySelectorAll() { return (selector) => collect(this, selector); }

  get querySelector() { return (selector) => collect(this, selector)[0] ?? null; }

  getElementsByTagName() { return (tag) => collect(this, String(tag).toLowerCase()); }

  getElementsByClassName() { return (name) => collect(this, `.${name}`); }

  getElementsByName() { return () => makeNodeList([]); }

  matches() { return false; }

  closest() { return null; }

  addEventListener() {}

  removeEventListener() {}

  dispatchEvent() { return false; }
}

class TextNode extends DomNode {
  constructor(doc, value) {
    super(doc, 3, "#text");
    this.data = String(value);
  }

  get textContent() { return this.data; }

  set textContent(value) { this.data = String(value); }

  get outerHTML() { return this.data; }
}

class CommentNode extends DomNode {
  constructor(doc, value) {
    super(doc, 8, "#comment");
    this.data = String(value);
  }

  get textContent() { return ""; }

  get outerHTML() { return `<!--${this.data}-->`; }
}

class DocumentType extends DomNode {
  constructor(doc) { super(doc, 10, "html"); }

  get outerHTML() { return "<!DOCTYPE html>"; }
}

class Element extends DomNode {
  constructor(doc, tagName) {
    super(doc, 1, String(tagName).toUpperCase());
    this.localName = String(tagName).toLowerCase();
    this.tagName = this.nodeName;
    this.nodeName = this.localName;
    this.attrs = {};
    this.style = createStyle();
    this.srcdoc = "";
    this.src = "";
    this.href = "";
    this.id = "";
    this.className = "";
    this.value = "";
    this.tabIndex = -1;
    this.dataset = {};
  }

  get attributes() { return Object.entries(this.attrs).map(([name, value]) => ({ name, value })); }

  get attributeNames() { return Object.keys(this.attrs); }

  hasAttribute(name) { return Object.prototype.hasOwnProperty.call(this.attrs, String(name).toLowerCase()); }

  hasAttributes() { return Object.keys(this.attrs).length > 0; }

  getAttribute(name) {
    const key = String(name).toLowerCase();
    return Object.prototype.hasOwnProperty.call(this.attrs, key) ? this.attrs[key] : null;
  }

  getAttributeNS(_ns, name) { return this.getAttribute(name); }

  setAttribute(name, value) { this.attrs[String(name).toLowerCase()] = String(value); }

  setAttributeNS(_ns, name, value) { this.setAttribute(name, value); }

  removeAttribute(name) { delete this.attrs[String(name).toLowerCase()]; }

  toggleAttribute(name, force) {
    const key = String(name).toLowerCase();
    const present = this.hasAttribute(key);
    const want = force === undefined ? !present : Boolean(force);
    if (want) this.attrs[key] = "";
    else delete this.attrs[key];
    return want;
  }

  get classList() {
    const self = this;
    const values = () => (self.className ? self.className.split(/\s+/).filter(Boolean) : []);
    return {
      add(name) { const list = values(); if (!list.includes(name)) list.push(name); self.className = list.join(" "); },
      remove(name) { self.className = values().filter((item) => item !== name).join(" "); },
      toggle(name) { if (self.classList.contains(name)) { self.classList.remove(name); return false; } self.classList.add(name); return true; },
      contains(name) { return values().includes(name); },
      item(index) { return values()[index] ?? null; },
      get length() { return values().length; },
    };
  }

  get outerHTML() {
    const attrs = Object.entries(this.attrs)
      .map(([name, value]) => (value === "" ? ` ${name}` : ` ${name}="${value}"`))
      .join("");
    const tag = this.localName;
    if (VOID_TAGS.has(tag)) return `<${tag}${attrs}>`;
    return `<${tag}${attrs}>${this.innerHTML}</${tag}>`;
  }

  get contentWindow() { return createFrameWindow(); }

  get contentDocument() { return this.contentWindow.document; }

  focus() {}

  blur() {}

  click() {}
}

class HTMLElement extends Element {}
class HTMLUnknownElement extends HTMLElement {}
class HTMLDivElement extends HTMLElement {}
class HTMLIFrameElement extends HTMLElement {}
class HTMLImageElement extends HTMLElement {}
class HTMLScriptElement extends HTMLElement {}
class HTMLStyleElement extends HTMLElement {}
class HTMLAnchorElement extends HTMLElement {}
class HTMLFormElement extends HTMLElement {}
class SVGElement extends Element {}
class MathMLElement extends Element {}
class MathMLElementSpecific extends MathMLElement {}
class NodeList {}
class HTMLCollection extends NodeList {}
class DOMTokenList {}

function makeNodeList(items) {
  const list = Object.create(NodeList.prototype);
  list.item = (index) => items[index] ?? null;
  list.namedItem = () => null;
  list.length = items.length;
  list[Symbol.iterator] = function* iterate() { yield* items; };
  return list;
}

NodeList.prototype.forEach = function forEach(callback, thisArg) {
  for (let i = 0; i < this.length; i += 1) callback.call(thisArg, this[i], i, this);
};
NodeList.prototype.entries = function* entries() { for (let i = 0; i < this.length; i += 1) yield [i, this[i]]; };
NodeList.prototype.keys = function* keys() { for (let i = 0; i < this.length; i += 1) yield i; };

const ELEMENT_CTOR = {
  div: HTMLDivElement,
  iframe: HTMLIFrameElement,
  img: HTMLImageElement,
  script: HTMLScriptElement,
  style: HTMLStyleElement,
  a: HTMLAnchorElement,
  form: HTMLFormElement,
  math: MathMLElement,
  svg: SVGElement,
  mglyph: MathMLElementSpecific,
  malignmark: MathMLElementSpecific,
};

class DocumentFragment extends Element {
  constructor(doc) {
    super(doc, "#document-fragment");
    this.nodeType = 11;
    this.nodeName = "#document-fragment";
  }

  get outerHTML() { return this.innerHTML; }
}

class Document extends DomNode {
  constructor() {
    super(null, 9, "#document");
    this.ownerDocument = this;
    this.readyState = "complete";
    this.referrer = "https://duck.ai/";
    this.title = "Duck.ai";
    this.cookie = "";
    this.domain = "duck.ai";
    this.characterSet = "UTF-8";
    this.compatMode = "CSS1Compat";
    this.doctype = new DocumentType(this);
    this.documentElement = new Element(this, "html");
    this.head = new Element(this, "head");
    this.body = new Element(this, "body");
    this.documentElement.appendChild(this.head);
    this.documentElement.appendChild(this.body);
    this.appendChild(this.documentElement);
  }

  createElement(tag) {
    const name = String(tag).toLowerCase();
    const Ctor = ELEMENT_CTOR[name] ?? (HTML_TAGS.has(name) ? HTMLElement : HTMLUnknownElement);
    return new Ctor(this, name);
  }

  createElementNS(_ns, tag) { return this.createElement(tag); }

  createTextNode(value) { return new TextNode(this, value); }

  createComment(value) { return new CommentNode(this, value); }

  createDocumentFragment() { return new DocumentFragment(this); }

  createEvent() { return { initEvent() {}, type: "", target: null, preventDefault() {}, stopPropagation() {} }; }

  getElementById() { return null; }

  getElementsByName() { return makeNodeList([]); }

  getElementsByClassName(name) { return collect(this, `.${name}`); }
}

function matchesSimple(element, selector) {
  const trimmed = String(selector).trim();
  if (trimmed === "" || trimmed === ":scope") return trimmed === ":scope" ? element === selector.__root : false;
  if (trimmed === "*") return true;
  for (const part of trimmed.split(",")) {
    const piece = part.trim();
    if (piece === "*") return true;
    const match = piece.match(/^([a-zA-Z][\w-]*|[\w-]+)?((?:\.[\w-]+)*)(?:\[([^\]]*)\])?$/);
    if (match === null) continue;
    const [, tag, classes, attrs] = match;
    if (tag && element.localName !== tag.toLowerCase()) continue;
    if (classes) {
      let ok = true;
      for (const cls of classes.split(".")) {
        if (cls && !element.classList.contains(cls)) { ok = false; break; }
      }
      if (!ok) continue;
    }
    if (attrs) {
      const attrMatch = attrs.match(/^([^\]=~|^$*]+)(?:([~|^$*]?=)["']?([^\]"']*)["']?)?$/);
      if (attrMatch) {
        const [, name, op, value] = attrMatch;
        if (!element.hasAttribute(name)) continue;
        if (op === "=" && element.getAttribute(name) !== value) continue;
      }
    }
    return true;
  }
  return false;
}

function collect(root, selector) {
  const results = [];
  const walk = (node) => {
    for (const child of node.childNodes) {
      if (child.nodeType !== 1) continue;
      if (matchesSimple(child, selector)) results.push(child);
      walk(child);
    }
  };
  walk(root);
  return makeNodeList(results);
}

function tokenize(html) {
  const tokens = [];
  const pattern = /<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<![^>]*>|<\/([a-zA-Z][^\s/>]*)\s*>|<([a-zA-Z][^\s/>]*)((?:"[^"]*"|'[^']*'|[^>"'])*?)(\/?)>/g;
  let index = 0;
  let match;
  while ((match = pattern.exec(html)) !== null) {
    if (match.index > index) tokens.push({ kind: "text", value: html.slice(index, match.index) });
    const raw = match[0];
    if (raw.startsWith("<!--")) tokens.push({ kind: "comment", value: raw.slice(4, raw.endsWith("-->") ? -3 : undefined) });
    else if (raw.startsWith("<!")) tokens.push({ kind: "doctype", value: raw });
    else if (match[1] !== undefined) tokens.push({ kind: "end", name: match[1].toLowerCase() });
    else tokens.push({ kind: "start", name: match[2].toLowerCase(), attrs: match[3] || "", selfClosing: match[4] === "/" });
    index = pattern.lastIndex;
  }
  if (index < html.length) tokens.push({ kind: "text", value: html.slice(index) });
  return tokens;
}

function parseAttributes(raw) {
  const attrs = {};
  const pattern = /([^\s=/]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'`=<>]+)))?/g;
  let match;
  while ((match = pattern.exec(raw)) !== null) {
    attrs[match[1].toLowerCase()] = decodeEntities(match[2] ?? match[3] ?? match[4] ?? "");
  }
  return attrs;
}

function parseFragment(doc, html) {
  const root = new Element(doc, "body");
  const stack = [root];
  const tokens = tokenize(html);
  let index = 0;

  const current = () => stack[stack.length - 1];
  const inForeign = () => FOREIGN_ROOTS.has(current().localName);
  const atIntegrationPoint = () => MATH_TEXT_INTEGRATION.has(current().localName) || current().localName === "annotation-xml";
  const openNames = () => stack.slice(1).map((node) => node.localName);
  const hasInScope = (name) => openNames().includes(name);
  const popUntil = (name) => {
    while (stack.length > 1) {
      if (stack.pop().localName === name) break;
    }
  };
  const removeFromStack = (name) => {
    const index = stack.map((node) => node.localName).lastIndexOf(name);
    if (index > 0) stack.splice(index, 1);
  };
  const generateImpliedEndTags = (except) => {
    while (stack.length > 1) {
      const tag = current().localName;
      if (!IMPLIED_END_TAGS.has(tag) || tag === except) break;
      stack.pop();
    }
  };

  while (index < tokens.length) {
    let token = tokens[index];
    index += 1;

    if (token.kind === "text") {
      if (token.value) current().appendChild(doc.createTextNode(decodeEntities(token.value)));
      continue;
    }
    if (token.kind === "comment") {
      current().appendChild(doc.createComment(token.value));
      continue;
    }
    if (token.kind === "doctype") continue;
    if (token.kind === "end") {
      const name = token.name;
      if (name === "br") {
        // "An end tag whose tag name is br": drop the attributes and act as if
        // this was a br start tag token.
        token = { kind: "start", name: "br", attrs: "", selfClosing: false };
      } else if ((RAW_TEXT_TAGS.has(current().localName) || RCDATA_TAGS.has(current().localName)) && current().localName !== name) {
        continue;
      } else if (name === "p" && !inForeign()) {
        if (!hasInScope("p")) {
          const synthesized = doc.createElement("p");
          current().appendChild(synthesized);
          stack.push(synthesized);
        }
        generateImpliedEndTags("p");
        if (current().localName === "p") stack.pop();
        continue;
      } else if (name === "form" && !inForeign()) {
        if (hasInScope("form")) {
          generateImpliedEndTags();
          removeFromStack("form");
        }
        continue;
      } else if (hasInScope(name)) {
        generateImpliedEndTags();
        popUntil(name);
        continue;
      } else {
        continue;
      }
    }

    const name = token.name;
    if (inForeign() && !atIntegrationPoint()) {
      if (MATH_TEXT_INTEGRATION.has(current().localName) && (name === "mglyph" || name === "malignmark")) {
        current().appendChild(doc.createElement(name));
        continue;
      }
      if (name === "svg") popUntil(current().localName);
    } else if (CLOSES_P.has(name) && hasInScope("p")) {
      generateImpliedEndTags("p");
      if (current().localName === "p") stack.pop();
    } else if (LIST_ITEM_TAGS.includes(name)) {
      if (hasInScope(name)) {
        generateImpliedEndTags(name);
        if (current().localName === name) stack.pop();
      }
    } else if (name === "option" || name === "optgroup") {
      if (current().localName === name) stack.pop();
    } else if (RCDATA_TAGS.has(name)) {
      if (current().localName === name) stack.pop();
    }

    const element = doc.createElement(name);
    Object.assign(element.attrs, parseAttributes(token.attrs));
    current().appendChild(element);
    if (VOID_TAGS.has(name) || token.selfClosing) continue;

    if ((RAW_TEXT_TAGS.has(name) || RCDATA_TAGS.has(name)) && !inForeign()) {
      const parts = [];
      while (index < tokens.length && !(tokens[index].kind === "end" && tokens[index].name === name)) {
        const next = tokens[index];
        parts.push(next.kind === "text" ? next.value : next.kind === "start" ? next.name : "");
        index += 1;
      }
      if (index < tokens.length) index += 1;
      if (parts.length) element.appendChild(doc.createTextNode(RAW_TEXT_TAGS.has(name) ? parts.join("") : decodeEntities(parts.join(""))));
      continue;
    }
    stack.push(element);
  }
  return root.childNodes;
}

const DEFAULT_COMPUTED = {
  display: "block",
  visibility: "visible",
  position: "static",
  "z-index": "auto",
  opacity: "1",
  overflow: "visible",
  "overflow-x": "visible",
  "overflow-y": "visible",
  "background-color": "rgba(0, 0, 0, 0)",
  color: "rgb(0, 0, 0)",
  "font-size": `${PX_PER_EM}px`,
  "font-family": "Times New Roman",
  "font-weight": "400",
  "line-height": "normal",
  margin: "0px",
  "margin-top": "0px",
  "margin-bottom": "0px",
  "margin-left": "0px",
  "margin-right": "0px",
  padding: "0px",
  "padding-top": "0px",
  "padding-bottom": "0px",
  "padding-left": "0px",
  "padding-right": "0px",
  border: "0px none rgb(0, 0, 0)",
  width: "auto",
  height: "auto",
  transform: "none",
  transition: "all 0s ease 0s",
  animation: "none",
  "content-visibility": "visible",
  "pointer-events": "auto",
  "user-select": "auto",
  "-webkit-font-smoothing": "auto",
};

const NORMALIZED_DEFAULT_TAGS = new Set(["div", "span", "p", "a", "b", "i", "em", "strong", "label", "button", "li", "ul", "h1", "h2", "h3", "h4", "h5", "h6"]);

function computedStyleOf(element) {
  const box = layoutOf(element);
  const computed = createStyle();
  for (const [name, value] of Object.entries(DEFAULT_COMPUTED)) computed.setProperty(name, value);
  const tag = element && element.localName ? element.localName : "div";
  computed.setProperty("display", NORMALIZED_DEFAULT_TAGS.has(tag) ? "block" : DEFAULT_COMPUTED.display);
  if (element && element.style && typeof element.style.getPropertyValue === "function") {
    for (const [name, value] of parseStyle(element.style.cssText)) computed.setProperty(name, value);
  }
  computed.setProperty("width", box.width ? `${box.width}px` : "auto");
  computed.setProperty("height", box.height ? `${box.height}px` : "auto");
  computed.setProperty("content-visibility", "visible");
  return computed;
}

let frameWindows = [];

function createFrameWindow() {
  const document = new Document();
  const win = {
    location: { origin: "https://duck.ai", href: "https://duck.ai/", protocol: "https:", host: "duck.ai", hostname: "duck.ai", pathname: "/", search: "", hash: "" },
    navigator: globalThis.navigator,
    setTimeout, clearTimeout, setInterval, clearInterval,
    Array, Object, String, Number, Boolean, JSON, Math, Date, Promise, Proxy,
    Symbol, RegExp, Error, TypeError, RangeError, SyntaxError, Map, Set,
    WeakMap, WeakSet, Uint8Array, Int8Array, Float64Array, ArrayBuffer,
    parseInt, parseFloat, isNaN, isFinite, encodeURIComponent,
    decodeURIComponent, escape, unescape,
    Node: DomNode, Element, HTMLElement, HTMLDivElement, HTMLIFrameElement,
    HTMLUnknownElement, SVGElement, MathMLElement, NodeList, HTMLCollection,
    addEventListener() {}, removeEventListener() {}, dispatchEvent: () => false,
    getComputedStyle: (element) => computedStyleOf(element),
    requestAnimationFrame: (cb) => setTimeout(() => cb(Date.now()), 0),
    cancelAnimationFrame: (handle) => clearTimeout(handle),
    performance: globalThis.performance,
  };
  win.window = win;
  win.self = win;
  win.top = win;
  win.parent = win;
  win.frames = win;
  win.globalThis = win;
  win.document = document;
  frameWindows.push(win);
  return win;
}

function installGlobals(userAgent) {
  const navigator = {
    userAgent,
    appVersion: userAgent.slice(8),
    appName: "Netscape",
    appCodeName: "Mozilla",
    platform: "Win32",
    product: "Gecko",
    productSub: "20030107",
    vendor: "Google Inc.",
    vendorSub: "",
    language: "en-US",
    languages: ["en-US", "en"],
    onLine: true,
    cookieEnabled: true,
    doNotTrack: null,
    webdriver: false,
    hardwareConcurrency: 16,
    deviceMemory: 8,
    maxTouchPoints: 0,
    pdfViewerEnabled: true,
    plugins: { length: 0, item: () => null, namedItem: () => null, refresh() {} },
    mimeTypes: { length: 0, item: () => null, namedItem: () => null },
    javaEnabled: () => false,
    sendBeacon: () => true,
    getBattery: () => new Promise(() => {}),
    connection: { effectiveType: "4g", rtt: 50, downlink: 10, saveData: false },
  };

  const document = new Document();

  const location = { origin: "https://duck.ai", href: "https://duck.ai/", protocol: "https:", host: "duck.ai", hostname: "duck.ai", pathname: "/", search: "", hash: "" };

  const define = (name, value) => {
    Object.defineProperty(globalThis, name, { value, writable: true, configurable: true, enumerable: false });
  };

  define("window", globalThis);
  define("self", globalThis);
  define("top", globalThis);
  define("parent", globalThis);
  define("frames", globalThis);
  define("document", document);
  define("navigator", navigator);
  define("location", location);
  define("screen", { width: 1920, height: 1080, availWidth: 1920, availHeight: 1040, colorDepth: 24, pixelDepth: 24, orientation: { type: "landscape-primary", angle: 0 } });
  define("innerWidth", 1920);
  define("innerHeight", 947);
  define("outerWidth", 1920);
  define("outerHeight", 1040);
  define("devicePixelRatio", 1);
  define("screenX", 0);
  define("screenY", 0);
  define("screenLeft", 0);
  define("screenTop", 0);
  define("scrollX", 0);
  define("scrollY", 0);
  define("pageXOffset", 0);
  define("pageYOffset", 0);
  define("deviceMemory", 8);
  define("hardwareConcurrency", 16);
  define("Node", DomNode);
  define("NodeList", NodeList);
  define("HTMLCollection", HTMLCollection);
  define("DOMTokenList", DOMTokenList);
  define("Element", Element);
  define("HTMLElement", HTMLElement);
  define("HTMLUnknownElement", HTMLUnknownElement);
  define("HTMLDivElement", HTMLDivElement);
  define("HTMLIFrameElement", HTMLIFrameElement);
  define("HTMLImageElement", HTMLImageElement);
  define("HTMLScriptElement", HTMLScriptElement);
  define("HTMLStyleElement", HTMLStyleElement);
  define("HTMLAnchorElement", HTMLAnchorElement);
  define("HTMLFormElement", HTMLFormElement);
  define("SVGElement", SVGElement);
  define("MathMLElement", MathMLElement);
  define("localStorage", makeStorage());
  define("sessionStorage", makeStorage());
  define("indexedDB", { open: () => ({ onsuccess: null, onerror: null, onupgradeneeded: null, result: null }) });
  define("requestAnimationFrame", (cb) => setTimeout(() => cb(Date.now()), 0));
  define("cancelAnimationFrame", (handle) => clearTimeout(handle));
  define("requestIdleCallback", (cb) => setTimeout(() => cb({ didTimeout: false, timeRemaining: () => 50 }), 0));
  define("cancelIdleCallback", (handle) => clearTimeout(handle));
  define("getComputedStyle", (element) => computedStyleOf(element));
  define("matchMedia", () => ({ matches: false, media: "", addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {}, dispatchEvent: () => false }));
  define("alert", () => {});
  define("confirm", () => false);
  define("prompt", () => null);
  define("open", () => null);
  define("close", () => {});
  define("focus", () => {});
  define("blur", () => {});
  define("scroll", () => {});
  define("scrollTo", () => {});
  define("scrollBy", () => {});
  define("postMessage", () => {});
  define("print", () => {});
  define("addEventListener", () => {});
  define("removeEventListener", () => {});
  define("dispatchEvent", () => false);
  define("atob", (value) => Buffer.from(String(value), "base64").toString("binary"));
  define("btoa", (value) => Buffer.from(String(value), "binary").toString("base64"));
  define("TextEncoder", require("util").TextEncoder);
  define("TextDecoder", require("util").TextDecoder);
  define("structuredClone", (value) => JSON.parse(JSON.stringify(value)));
  define("performance", { now: () => Number(process.hrtime.bigint() / 1000n) / 1000, timeOrigin: Date.now(), timing: {}, getEntriesByType: () => [], mark() {}, measure() {}, clearMarks() {}, clearMeasures() {}, toJSON: () => ({}) });
  const nodeCrypto = require("crypto");
  define("crypto", {
    getRandomValues: (array) => { nodeCrypto.randomFillSync(array); return array; },
    randomUUID: () => nodeCrypto.randomUUID(),
    subtle: nodeCrypto.webcrypto.subtle,
  });
  define("Image", HTMLImageElement);
  define("Option", Element);
  define("FormData", globalThis.FormData);
  Object.defineProperty(globalThis, Symbol.toStringTag, { value: "Window", configurable: true });
  frameWindows = [];
}

function readStdin() {
  return new Promise((resolve) => {
    const chunks = [];
    process.stdin.on("data", (chunk) => chunks.push(chunk));
    process.stdin.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    process.stdin.resume();
  });
}

async function evaluateSource(source, userAgent) {
  installGlobals(userAgent);
  const context = vm.createContext(globalThis, { name: "duckai-jsa" });
  const script = new vm.Script(`(function(){ const __r = (${source}); return typeof __r === "function" ? __r() : __r; })()`, { filename: "jsa.js" });
  return script.runInContext(context, { timeout: 15000, displayErrors: true });
}

async function evaluateScript(source, userAgent) {
  const result = await evaluateSource(source, userAgent);
  if (result === null || typeof result !== "object" || !Array.isArray(result.client_hashes)) {
    throw new Error(`unexpected attestation shape: ${JSON.stringify(result)}`);
  }
  if (!Array.isArray(result.server_hashes) || result.meta === null || typeof result.meta !== "object") {
    throw new Error("attestation is missing server_hashes or meta");
  }
  return result;
}

async function main() {
  const input = await readStdin();
  let payload;
  try {
    payload = JSON.parse(input);
  } catch (error) {
    process.stdout.write(JSON.stringify({ ok: false, error: "malformed input" }));
    return;
  }
  if (typeof payload.script !== "string" || !payload.script) {
    process.stdout.write(JSON.stringify({ ok: false, error: "missing script" }));
    return;
  }
  const userAgent = typeof payload.user_agent === "string" && payload.user_agent ? payload.user_agent : "";
  try {
    process.stdout.write(JSON.stringify({ ok: true, result: await evaluateScript(payload.script, userAgent) }));
  } catch (error) {
    const message = error && error.message ? String(error.message) : String(error);
    process.stdout.write(JSON.stringify({ ok: false, error: message.slice(0, 400) }));
  }
}

if (typeof module !== "undefined" && module.exports) {
  if (require.main === module) {
    main();
  } else {
    module.exports = { evaluateScript, evaluateSource };
  }
} else {
  main();
}
