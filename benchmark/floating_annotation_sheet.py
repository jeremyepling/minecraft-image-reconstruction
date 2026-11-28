"""Render the round-three annotation sheet for mobile."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont



REVIEW = Path(__file__).parent/"reviews/floating-grid-round-04"

WIDTH = 1600
MARGIN = 62
BODY_WIDTH = WIDTH-2*MARGIN


def font(size, bold=False):
    return ImageFont.truetype("arialbd.ttf" if bold else "arial.ttf",size)


def wrap(draw, text, face):
    lines = [""]
    for word in text.split():
        joined = f"{lines[-1]} {word}".strip()
        if draw.textlength(joined,font=face) > BODY_WIDTH:
            lines.append(word)
        else:
            lines[-1] = joined
    return lines


def main():
    sheet = Image.new("RGB",(WIDTH,4600),"#14202b")
    draw = ImageDraw.Draw(sheet)
    y = 52

    def line(text,size=32,bold=False,color="#edf3f7"):
        nonlocal y
        face = font(size,bold)
        for segment in wrap(draw,text,face):
            draw.text((MARGIN,y),segment,fill=color,font=face)
            y += size+10

    def picture(path):
        nonlocal y
        with Image.open(path) as source:
            height = round(source.height*BODY_WIDTH/source.width)
            sheet.paste(source.convert("RGB").resize((BODY_WIDTH,height),Image.Resampling.LANCZOS),(MARGIN,y))
        y += height+24

    def section(title,guess,path,prompt):
        nonlocal y
        y += 26
        draw.line((MARGIN,y,WIDTH-MARGIN,y),fill="#506779",width=2)
        y += 34
        line(title,49,True)
        line(guess,33)
        y += 14
        picture(path)
        line(prompt,34,True,"#ffe093")

    line("Floating builds · grid review",67,True)
    line("Annotate this image or reply with the prompt ID. Mark the first wrong edge or cell. For a grid, rate coverage, alignment, and block size separately.",34)
    section("T1  Twinspire · brick beside sandstone",
            "Cyan shows the new grid. Green marks visible brick pixels supporting added cells. Gold shows recorded visible face edges. The exported build still drops this brick wall.",
            REVIEW/"twinspire.png",
            "T1  Mark fragments that should count as a brick. Circle any cyan cell that spans two bricks. Mark a visible edge that could establish how this wall connects.")
    section("M1  Riverside Mill · roof against cloud",
            "Red outlines five extra placements mostly on cloud. Green outlines a correct block that touches sky. Orange marks sky pixels inside the detected wall mask.",
            REVIEW/"mill.jpg",
            "M1  Trace the true roof edge where it meets cloud. Circle pale roof material to preserve. Flag any roof gap or extra cell outside the red boxes.")
    section("V1  Viaduct · left pier, second base, and upper posts",
            "Blue shows the attached left column. Turquoise marks the refitted second base. Orange marks the new two-block post grids.",
            REVIEW/"viaduct-visible-extents.jpg",
            "V1  Check the second base width. Mark posts that look too tall, low, or misplaced in depth. Flag where a grid continues through an opening.")
    y += 35
    line("For each case, report only what you can see.",33,True)
    line("Example reply · T1 coverage missing one partial face, alignment one seam too far left, block size right.",28,color="#c9d8e4")
    assert y+45 < sheet.height
    sheet.crop((0,0,WIDTH,y+45)).save(REVIEW/"mobile-annotation.png",optimize=True)


if __name__=="__main__":
    main()

