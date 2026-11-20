"""Render a self-contained dashboard from the same tables exported to Power BI."""
import json
from pathlib import Path


def render(tables, destination, counts):
    data = json.dumps({name: json.loads(table.to_json(orient="records"))
                       for name, table in tables.items()}, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("dashboard_template.html")
    destination.write_text(
        template.read_text().replace("{{DATA}}", data).replace(
            "{{SPLITS}}", " / ".join(f"{n:,}" for n in counts)
        ), encoding="utf-8"
    )
