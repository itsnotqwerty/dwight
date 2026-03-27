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

  /**
   * Convert a plain text string into formatted HTML.
   * - Escapes HTML entities first
   * - Wraps '>' lines in a greentext span
   * - Applies inline markdown
   *
   * @param {string} rawText
   * @returns {string} HTML string safe to assign to innerHTML
   */
  function renderText(rawText) {
    return rawText
      .split('\n')
      .map(function (line) {
        const escaped = escapeHtml(line);
        // Greentext: line starts with '>' (after optional whitespace)
        if (/^\s*&gt;/.test(escaped)) {
          return '<span class="greentext">' + applyInlineMarkdown(escaped) + '</span>';
        }
        return applyInlineMarkdown(escaped);
      })
      .join('\n');
  }

  /**
   * Render formatted text into a <pre> element.
   * Sets innerHTML and adds the dwight-rendered class.
   *
   * @param {HTMLElement} el      Target element (typically a <pre>)
   * @param {string}      rawText Plain text to render
   */
  function applyToElement(el, rawText) {
    el.classList.add('dwight-rendered');
    el.innerHTML = renderText(rawText);
  }

  global.DwightFormat = { escapeHtml, renderText, applyToElement };
})(window);
