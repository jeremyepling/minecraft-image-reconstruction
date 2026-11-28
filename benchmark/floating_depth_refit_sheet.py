"""Draw the three reviewed relief cases on their source images from the scored grids."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmark.floating_depth_refit import EXTENDED, GUARDED, PRIOR, REVIEW, UNIT, paths, project, read


def panel_grid(im, h, cells, color):
    draw = ImageDraw.Draw(im)
    for uv in cells:
        corners = project(np.asarray(h),UNIT+uv)
        points = list(map(tuple,corners))
        draw.line(points+[points[0]],fill=color,width=2)


def main():
    cases = [("holdout4","train-riverside-mill-1-warm-low-left","Riverside Mill"),
             ("holdout4","train-viaduct-01-riverstone-low-left","Viaduct"),
             ("original","test-castle-01-ashlar-high-right","Castle")]
    rows = {(r["panel"],r["id"]):r for r in read(REVIEW/"conditions.json.gz")}
    font = ImageFont.truetype("arial.ttf",19)
    small = ImageFont.truetype("arial.ttf",16)
    for panel,name,title in cases:
        case = next(c for c in paths(panel)[0][3] if c["id"]==name)
        src = Image.open(case["variants"]["sky"]["image"]).convert("RGB")
        row = rows[panel,name]
        parent = read(PRIOR/panel/"source/charts/sky"/name/"charts.json")
        chart = next(c for c in parent["charts"] if c["id"]==row["patches"][0]["chart"])
        all_uv = [[c["u"],c["v"]] for c in chart["cells"]]
        xy = project(np.array(chart["H"]),np.asarray(all_uv)[:,None,:]+UNIT).reshape(-1,2)
        box = (*np.maximum([0,0],np.floor(xy.min(0))-20).astype(int),
               *np.minimum(src.size,np.ceil(xy.max(0))+20).astype(int))
        canvas = Image.new("RGB",(1760,520),"#161d24")
        draw = ImageDraw.Draw(canvas)
        for i,(arm,label) in enumerate((("source-depth","Prior depth"),(EXTENDED,"Local grids and small steps"),
                                        (GUARDED,"Signed boundary check"),
                                        ("native-face-refit","Recorded face corners"))):
            im = src.copy()
            for part in row["patches"]:
                if arm=="source-depth":
                    panel_grid(im,part["parent_H"],part["old_uv"],"#ffc950")
                elif arm==GUARDED and not part["boundary_guard"]["accept"]:
                    panel_grid(im,part["parent_H"],part["old_uv"],"#c7c7ce")
                else:
                    fit = part["source-refit" if arm in (EXTENDED,GUARDED) else arm]
                    panel_grid(im,fit["H"],fit["uv"],"#ffc950")
                    anchor = part["reference_anchor"]
                    if arm=="native-face-refit" and anchor:
                        ring = list(map(tuple,anchor["corners_px"]))
                        ImageDraw.Draw(im).line(ring+ring[:1],fill="#ffffff",width=5)
            if arm==EXTENDED or (arm==GUARDED and any(p["boundary_guard"]["accept"] for p in row["patches"])):
                for part in row["extra_patches"]:
                    panel_grid(im,part["H"],part["uv"],"#4feada")
            view = im.crop(box)
            view.thumbnail((428,386),Image.Resampling.LANCZOS)
            x = i*440
            canvas.paste(view,(x+(440-view.width)//2,64+(386-view.height)//2))
            draw.text((x+10,36),label,font=font,fill="white")
            counts = row["score"][arm]["geometry"]["strict"]
            draw.text((x+10,453),f'{counts["correct"]} correct   {counts["extras"]} extra',font=small,fill="white")
        draw.text((10,7),title,font=font,fill="white")
        draw.text((10,488),"Gold   original relief patch     |     Teal   additional depth steps     |     Gray   kept on parent plane     |     White   recorded anchor face",
                  font=small,fill="white")
        canvas.save(REVIEW/(name+".jpg"),quality=92)


if __name__=="__main__":
    main()
