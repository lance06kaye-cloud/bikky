"""Design tokens: the one place for the app's colors, type sizes and spacing.

The charts, the browser build and the injected stylesheet all read from here.
.streamlit/config.toml can't import Python, so a test checks it matches THEME.
"""

# Colors. One accent (ACCENT) for primary buttons, links and the promo group;
# everything else is neutral. CONTROL marks the comparison group in charts only.
SURFACE = "#fcfcfb"
SURFACE_RAISED = "#f3f2ee"
TEXT = "#0b0b0b"
MUTED_TEXT = "#52514e"  # 7.9:1 on SURFACE
FAINT_TEXT = "#6b6a66"  # 5.3:1 on SURFACE, still AA for body text
GRID = "#e8e7e3"
ACCENT = "#226dc6"  # 5.0:1 on SURFACE, 5.2:1 under white button text
CONTROL = "#c9541f"
ACCENT_WASH = "rgba(34, 109, 198, 0.10)"
NEUTRAL_WASH = "rgba(82, 81, 78, 0.07)"

# Type scale in px: six sizes
TYPE = {"xs": 12, "sm": 14, "md": 16, "lg": 20, "xl": 28, "xxl": 40}

# Spacing scale in px, on a 4px grid
SPACE = {1: 4, 2: 8, 3: 12, 4: 16, 6: 24, 8: 32, 12: 48}

RADIUS = 8

# Motion: short, for feedback only, and off for viewers who ask for less motion
MOTION_MS = 180

# Streamlit's theme, for .streamlit/config.toml and the browser build
THEME = {
    "base": "light",
    "primaryColor": ACCENT,
    "backgroundColor": SURFACE,
    "secondaryBackgroundColor": SURFACE_RAISED,
    "textColor": TEXT,
}


def css() -> str:
    """The stylesheet the app injects: tokens as CSS variables, plus the few rules
    Streamlit's theme can't express."""
    tokens = "\n".join(
        [f"  --color-surface: {SURFACE};", f"  --color-text: {TEXT};",
         f"  --color-muted: {MUTED_TEXT};", f"  --color-faint: {FAINT_TEXT};",
         f"  --color-accent: {ACCENT};", f"  --radius: {RADIUS}px;",
         f"  --motion: {MOTION_MS}ms;"]
        + [f"  --text-{name}: {size}px;" for name, size in TYPE.items()]
        + [f"  --space-{step}: {size}px;" for step, size in SPACE.items()]
    )
    return f"""<style>
:root {{
{tokens}
}}
h1, h2, h3, h4, .headline-key {{ text-wrap: balance; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] p,
.headline-detail {{ text-wrap: pretty; max-width: 72ch; }}
[data-testid="stMetricValue"], [data-testid="stTable"], .headline-key, .headline-detail {{
  font-variant-numeric: tabular-nums;
}}
/* The page title steps back so each view's own result is the largest thing on it */
[data-testid="stHeading"] h1 {{ font-size: var(--text-xl) !important; padding-bottom: var(--space-1); }}
.headline-key {{
  font-size: var(--text-xxl); font-weight: 700; line-height: 1.15;
  margin: var(--space-2) 0 var(--space-2);
}}
.headline-detail {{
  font-size: var(--text-md); color: var(--color-muted); line-height: 1.5;
  margin: 0 0 var(--space-6);
}}
@media (max-width: 640px) {{ .headline-key {{ font-size: var(--text-xl); }} }}
.faint {{ color: var(--color-faint); }}
button {{
  transition: background-color var(--motion) ease, border-color var(--motion) ease,
              color var(--motion) ease, transform var(--motion) ease;
}}
button:active:not(:disabled) {{ transform: scale(0.98); }}
@media (prefers-reduced-motion: reduce) {{
  button {{ transition: none; }}
  button:active:not(:disabled) {{ transform: none; }}
}}
</style>"""
