/**
 * DwightFormat — shared text formatting utilities.
 *
 * Features:
 *   - Greentext: lines starting with '>' are coloured green (4chan style)
 *   - Inline markdown: **bold**, *italic*, `code`
 *   - HTML escaping to prevent injection
 */
(function (global) {
  'use strict';

  // Inject stylesheet once
  const STYLE_ID = 'dwight-format-style';
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = [
      '.greentext { color: #789922; }',
      'pre.dwight-rendered { white-space: pre-wrap; word-break: break-word; }',
    ].join('\n');
    document.head.appendChild(style);
  }

  /**
   * Escape special HTML characters in a plain string.
   * @param {string} str
   * @returns {string}
   */
  function escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /**
   * Apply inline markdown formatting to an already-HTML-escaped string.
   * Handles **bold**, *italic*, and `inline code`.
   * Order matters: code spans are processed first so their content is not
   * further interpreted as bold/italic markers.
   * @param {string} escaped  HTML-escaped text (no raw < or >)
   * @returns {string}
   */
  function applyInlineMarkdown(escaped) {
    // `inline code` — match backtick-delimited spans
    escaped = escaped.replace(/`([^`]+)`/g, '<code>$1</code>');
    // **bold**
    escaped = escaped.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    // *italic* (single asterisk, not doubled)
    escaped = escaped.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    return escaped;
  }

  // Words whose inner letters are replaced with '*' when censoring is on.
  // Checked case-insensitively against whole words only.
  const BLACKLIST = new Set([
    'nigger', 'nigga',
    'fag',
    'kike',
    'tranny', 'trannies',
    'chink',
    'spic',
    'wetback',
    'gook',
    'cunt',
		'cuck',
		'retard',
    'whore',
    'dyke',
  ]);

  /**
   * Censor a single word, keeping its first and last character and replacing
   * all inner characters with asterisks.  Words of 1–2 characters are left
   * unchanged.
   * @param {string} word  Original word (mixed case OK)
   * @returns {string}
   */
  function _censorWord(word) {
    if (word.length <= 2) return word;
    return word[0] + '*'.repeat(word.length - 2) + word[word.length - 1];
  }

  /**
   * Censor blacklisted words inside an HTML string, leaving tag markup
   * completely untouched.  The alternation `(<[^>]+>)|(\b[A-Za-z]+\b)`
   * means every match is either:
   *   - An HTML tag  → returned as-is (first capture group)
   *   - A plain word → censored if blacklisted (second capture group)
   *
   * This must be called AFTER markdown rendering so that the asterisks
   * inserted by `_censorWord` are never seen by the markdown processor
   * and cannot accidentally trigger bold/italic formatting.
   *
   * @param {string} html  HTML string (output of applyInlineMarkdown)
   * @returns {string}     HTML string with blacklisted words censored
   */
  /**
   * Return true if any blacklisted term appears anywhere inside *word*.
   * @param {string} word  lower-cased word
   * @returns {boolean}
   */
  function _isBlacklisted(word) {
    for (const term of BLACKLIST) {
      if (word.includes(term)) return true;
    }
    return false;
  }

  function censorText(html) {
    return html.replace(/(<[^>]*>)|([A-Za-z]+)/g, function (match, tag, word) {
      if (tag !== undefined) return tag;  // HTML tag — pass through unchanged
      return _isBlacklisted(word.toLowerCase()) ? _censorWord(word) : word;
    });
  }

  /**
   * Convert a plain text string into formatted HTML.
   * Pipeline per line:
   *   1. HTML-escape raw text
   *   2. Apply inline markdown  (**bold**, *italic*, `code`)
   *   3. If censoring: replace blacklisted words (after markdown, so `*`
   *      chars from censoring are never re-interpreted as markdown syntax)
   *   4. Wrap greentext lines in a <span>
   *
   * @param {string}  rawText
   * @param {boolean} [censor=false]
   * @returns {string} HTML string safe to assign to innerHTML
   */
  function renderText(rawText, censor) {
    return rawText
      .split('\n')
      .map(function (line) {
        const escaped = escapeHtml(line);
        const markdown = applyInlineMarkdown(escaped);
        const text = censor ? censorText(markdown) : markdown;
        // Greentext: line starts with '>' (after optional whitespace)
        if (/^\s*&gt;/.test(escaped)) {
          return '<span class="greentext">' + text + '</span>';
        }
        return text;
      })
      .join('\n');
  }

  /**
   * Render formatted text into a <pre> element.
   * Sets innerHTML and adds the dwight-rendered class.
   *
   * @param {HTMLElement} el      Target element (typically a <pre>)
   * @param {string}      rawText Plain text to render
   * @param {boolean}     [censor=false] Whether to censor blacklisted words
   */
  function applyToElement(el, rawText, censor) {
    el.classList.add('dwight-rendered');
    el.innerHTML = renderText(rawText, censor);
  }

  global.DwightFormat = { escapeHtml, censorText, renderText, applyToElement };
})(window);
