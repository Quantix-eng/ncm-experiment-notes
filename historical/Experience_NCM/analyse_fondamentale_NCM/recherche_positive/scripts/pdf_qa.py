"""Render every publication page and make contact sheets for visual review."""
from base import *
sys.path.insert(0,str(ROOT/'generalisation_architecturale/.deps'))
import pymupdf
from PIL import Image,ImageOps,ImageDraw
def main():
    path=D/'PAPER.pdf';doc=pymupdf.open(path);out=D/'.cache/pdf_preview';out.mkdir(parents=True,exist_ok=True);thumbs=[];pages=[]
    for i,page in enumerate(doc):
        text=page.get_text();assert '\ufffd' not in text
        pix=page.get_pixmap(matrix=pymupdf.Matrix(1.3,1.3),alpha=False);p=out/f'page_{i+1:02d}.png';pix.save(p)
        im=Image.open(p).convert('RGB');im.thumbnail((390,550));canvas=Image.new('RGB',(410,580),'#dddddd');canvas.paste(im,((410-im.width)//2,15));ImageDraw.Draw(canvas).text((10,562),f'Page {i+1}',fill='black');thumbs.append(canvas);pages.append(dict(page=i+1,width=page.rect.width,height=page.rect.height,text_characters=len(text),render=str(p.relative_to(D))))
    for lo in range(0,len(thumbs),6):
        sheet=Image.new('RGB',(1230,1160),'#bbbbbb')
        for j,im in enumerate(thumbs[lo:lo+6]):sheet.paste(im,((j%3)*410,(j//3)*580))
        sheet.save(out/f'contact_{lo//6+1:02d}.png')
    latex=(D/'logs/latex.log').read_text();assert 'Overfull \\hbox' not in latex and 'Overfull \\vbox' not in latex
    write(D/'results/pdf_inspection.json',dict(pdf_sha256=sha(path),n_pages=len(doc),pages=pages,automated_checks='All pages rendered; no replacement glyphs; no overfull LaTeX boxes.',visual_review='Contact sheets and full-resolution pages must be reviewed by the researcher; script alone does not certify visual layout.'))
    log('pdf_rendered',pages=len(doc))
if __name__=='__main__':main()
