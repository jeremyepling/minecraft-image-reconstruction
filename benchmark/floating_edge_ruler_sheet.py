"""Show fitted source lines and the exact tiles used to verify a replacement grid."""
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from benchmark.floating_edge_aligned_ruler import OUT,REVIEW,PREFIX,UNIT,project,read,work


def main():
    rows=[r for r in read(REVIEW/"conditions.json.gz") if r["charts"]]
    rows.sort(key=lambda r:("stable" not in r["id"],"low-left" not in r["id"]))
    jobs={(g,c["id"]):c for g,c,_ in work()}
    sheet=Image.new("RGB",(1320,285*len(rows)),"white")
    draw=ImageDraw.Draw(sheet)
    for i,row in enumerate(rows):
        n,g=row["id"],row["panel"]
        image=Image.open(jobs[g,n]["variants"]["sky"]["image"]).convert("RGB")
        chart=next(iter(row["charts"].values()))
        lines=image.copy()
        ink=ImageDraw.Draw(lines)
        for family,color in zip(chart["lines"],("#b841e7","#f59b11")):
            for l,keep in zip(family.get("lines",[]),family.get("inliers",[])):
                if keep:
                    ink.line(list(map(tuple,l["ends"])),fill=color,width=3)
        grid=Image.open(OUT/g/"fine-material/assembly"/PREFIX/n/"source-overlay.jpg").convert("RGB")
        proof=image.copy()
        ink=ImageDraw.Draw(proof)
        fine=chart["refined"]
        h=np.asarray(fine["homography"])
        pts=[]
        for r in fine["local_repeat_check"].get("tests",[]):
            pts.append((r["uv"],"#42f1ac" if r["ok"] else "#ff746c"))
        for uv,color in pts:
            pts_screen=project(h,UNIT+uv)
            ink.line([tuple(p) for p in [*pts_screen,pts_screen[0]]],fill=color,width=4)
        extents=np.vstack([l["ends"] for f in chart["lines"] for l in f.get("lines",[])])
        box=tuple(map(int,(*np.maximum(0,extents.min(0)-25),*np.minimum(image.size,extents.max(0)+25))))
        draw.text((8,285*i+4),n.replace("train-",""),fill="black",font=ImageFont.truetype("arial.ttf",20))
        for j,(im,title) in enumerate(zip((lines,grid,proof),("U and V structural lines","Forced refined grid","Whole-sprite harmonic checks"))):
            im=im.crop(box)
            im.thumbnail((430,207),Image.Resampling.LANCZOS)
            sheet.paste(im,(440*j+(440-im.width)//2,285*i+44))
            draw.text((8+j*440,285*i+259),title,fill="black",font=ImageFont.truetype("arial.ttf",16))
    sheet.save(REVIEW/"structural-lines-and-sprites.jpg",quality=92)


if __name__=="__main__":
    main()
