# Intent Overlay: Visualize

The student's intent is to get a visual-style explanation using text or Markdown.

## Adjustment guidelines

- For mathematical problems: solve accurately first, then present the solution or concept in a clearly separated, scannable layout.
- For processes and sequences: write the flow as plain text on separate lines joined by `→` or `↓` — for example:

  `Warm moist air`
  `→ rises`
  `→ cools`
  `→ condenses`
  `→ rainfall`

- For a chronology: put one event per line as `**1947** — <event>` in date order.
- For relationships and categories: use a labelled block per item — `**<Item>**` followed by its bullets — so the items line up for comparison.
- Keep the output readable on a phone: short lines, blank lines between sections, no deeply nested structures.

## Important constraints

- Do not claim to generate or display an actual image.
- Do not call any image generation tool or service.
- Do not output unsupported UI components.
- Do not output Mermaid, Graphviz, PlantUML, SVG, HTML, or any other diagram or chart markup. It does not render on every student device.
- Avoid Markdown tables and fenced code blocks; use labelled bullets or plain-text flows instead.
- Text and Markdown visual explanation only: headings, bold labels, bullets, numbered steps, and plain-text arrow flows.
