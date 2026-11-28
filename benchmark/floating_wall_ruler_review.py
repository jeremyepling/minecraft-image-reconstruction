"""Render a compact phone-ready sheet for wall-grid round receipts."""
from pathlib import Path

from PIL import Image, ImageDraw

from benchmark.floating_corner_overwrite_review import BG, GOLD, MUTED, TEXT, draw_wrap, font, picture
from benchmark.floating_grid_signoff import make
from benchmark.floating_second_signoff import option


def grid_sheet(title, summary, cases, output):
    cards = []
    for item in cases:
        panel, name, wall = (item[key] for key in ("panel", "id", "wall"))
        relative = Path("sky")/name/"charts.json"
        source_options = [option(root,relative,wall) for root in item["roots"]]
        if item.get("export_cells"):
            chart = source_options[1][0][0]
            cells = chart.get("export_cells",chart["cells"])
            if "export_uv" in chart:
                allowed = set(map(tuple, chart["export_uv"]))
                cells = [c for c in cells if (c["u"],c["v"]) in allowed]
            source_options[1][0][0] = dict(chart,cells=cells)
        record = make(("grid",panel,name,wall,item["title"],item["note"]),dict(charts=source_options,scores=item["scores"]))
        card = Image.new("RGB",(1540,1500),BG)
        d = ImageDraw.Draw(card)
        d.text((38,14),record["title"],font=font(43,True),fill=TEXT)
        y = draw_wrap(d,(38,73),record["proposal"],1460,29)
        context = picture(record["context"])
        context.thumbnail((1120,300),Image.Resampling.LANCZOS)
        card.paste(context,((card.width-context.width)//2,y+10))
        y += context.height + 27
        captions = [*item["captions"],"Recorded visible edges"]
        images = [picture(o["images"][0]) for o in record["options"]] + [picture(record["gold"])]
        for i,(caption,im) in enumerate(zip(captions,images,strict=True)):
            x = 38 + 490*i
            d.text((x,y),caption,font=font(28,True),fill=TEXT)
            im.thumbnail((463,520),Image.Resampling.LANCZOS)
            card.paste(im,(x+(463-im.width)//2,y+49))
        y += max(im.height for im in images) + 68
        a,b = (o["score"] for o in record["options"])
        y = draw_wrap(d,(38,y),f"Whole export · {a[0]} → {b[0]} correct · {a[1]} → {b[1]} extras · {a[2]} → {b[2]} exact position and material matches.",1460,28,GOLD,True)
        cards.append(card.crop((0,0,card.width,y+28)))
    sheet = Image.new("RGB",(1540,190+sum(c.height+10 for c in cards)),BG)
    d = ImageDraw.Draw(sheet)
    d.text((38,23),title,font=font(55,True),fill=TEXT)
    y = draw_wrap(d,(38,100),summary,1460,28,MUTED)+30
    for card in cards:
        d.line((38,y,1502,y),fill="#567080",width=2)
        sheet.paste(card,(0,y+8))
        y += card.height+10
    sheet.crop((0,0,sheet.width,y)).save(output,optimize=True)
