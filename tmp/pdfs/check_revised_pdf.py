from pathlib import Path
import json
import re

import pdfplumber
from PIL import Image, ImageDraw
from pypdf import PdfReader

root = Path('/home/guillem/fragile')
work = root / 'tmp/pdfs'
pdf = work / 'volume_2_qft_yang_mills_proof_roadmap.pdf'
source = root / 'output/pdf/volume_2_qft_yang_mills_proof_roadmap.tex'
source_text = source.read_text()
expected_refs = sorted(
    int(n) for n in re.findall(r'\\textbf\{\[R(\d+)\]', source_text)
)
assert expected_refs == list(range(1, len(expected_refs) + 1))
with pdfplumber.open(pdf) as document:
    texts = [page.extract_text() or '' for page in document.pages]
    full = '\n'.join(texts)
    assert '\ufffd' not in full
    assert 'blocker' not in full.lower()
    assert 'packet mass' not in full.lower()
    assert len(re.findall(r'(?m)^Proof mechanism$', full)) == 18
    assert 'Poincar' in full
    assert 'Density correction is part of the existing construction' in full
    assert 'The finite algorithm already has field equations' in full
    assert 'The equilibrium correlator route is already specified' in full
    assert 'Equilibrium thermodynamics and the field state' in full
    assert 'Thermal balance and the complete population law' in full
    assert 'Population thermodynamics supplies uniform coercivity' in full
    assert 'Thermodynamic response is also explicit' in full
    references = sorted(set(int(n) for n in re.findall(r'\[R(\d+)\]', full)))
    assert references == expected_refs, references
    page_index = []
    for i, text in enumerate(texts, 1):
        lines = text.splitlines()
        page_index.append({'page': i, 'characters': len(text), 'opening': ' / '.join(lines[:3])})
    (work / 'revised_pdf_extracted.txt').write_text(full)
    (work / 'revised_page_index.json').write_text(json.dumps(page_index, indent=2))
    for needle in ['Density correction is part', 'A second continuum estimate', 'How the proved walker', 'Three spatial coordinates', 'Which reflection and symmetry', 'The finite algorithm already has field', 'The estimates that control']:
        print(needle, [i for i, text in enumerate(texts, 1) if needle in text])

reader = PdfReader(pdf)
print('Pages:', len(reader.pages))
print('Internal link annotations:', sum(len(page.get('/Annots', [])) for page in reader.pages))
print('Source results:', len(references))

files = sorted(work.glob('revision_page-*.png'))
assert len(files) == len(texts)
thumb_width = 340
thumb_height = 482
for offset in range(0, len(files), 9):
    sheet = Image.new('RGB', (3 * (thumb_width + 20), 3 * (thumb_height + 36)), '#e6ebee')
    draw = ImageDraw.Draw(sheet)
    for j, file in enumerate(files[offset:offset + 9]):
        page = Image.open(file).convert('RGB')
        page.thumbnail((thumb_width, thumb_height))
        x = (j % 3) * (thumb_width + 20) + 10
        y = (j // 3) * (thumb_height + 36) + 26
        sheet.paste(page, (x, y))
        draw.text((x, y - 18), f'Page {offset + j + 1}', fill='#17384a')
    out = work / f'revision_contact_{offset // 9 + 1}.png'
    sheet.save(out)
    print(out)
