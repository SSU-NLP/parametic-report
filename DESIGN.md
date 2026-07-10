# Parametic Report Design System

Reference: Airtable-inspired structured data workspace from getdesign.md.

## Visual Direction

Parametic Report uses a spreadsheet-database hybrid interface: dense records, colorful status accents, clear side navigation, and a right-side record detail panel. The product is an operational research tool, so the first screen is the usable analysis workspace, not a marketing page.

## Color

- Background: `#f7f8fb` with a subtle grid pattern.
- Surface: `#ffffff` and `#f2f4f8` for table headers and grouped controls.
- Text: `#172033`, muted text `#687385`.
- Accent set: blue `#2d7ff9`, green `#20c933`, yellow `#ffbf00`, red `#f82b60`.
- Use multiple accents as record/status markers, not full-page gradients.

## Layout

- Left sidebar for workspace navigation.
- Sticky topbar for view tabs and primary actions.
- Main area is a structured grid/table of analysis requests.
- Right panel shows the selected record with status, cache key, artifact root, and artifact list.
- Cards use small radii, 8px or less.

## Components

- Tabs are compact segmented controls.
- Inputs/selects are table-adjacent controls, not large hero forms.
- Primary action is green for `Run analysis`.
- Status badges are pill-shaped and compact.
- Artifact paths use monospace text.

## Responsive Behavior

- Desktop keeps sidebar + grid + record panel.
- Under 980px, hide sidebar and stack the record panel below the grid.
- Tables collapse to the first two columns on small screens.
