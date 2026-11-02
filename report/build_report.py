"""Builds the TM471B report from the university's Word template, so the template's styles, headers and page numbers stay.

Run with a Python that has lxml:   python report/build_report.py
Then report/update_word.ps1 lets Word refresh the contents pages and saves a PDF for checking.

The chapters' text, tables and figures are in report/content.py. The template itself is never changed.
"""
import copy
import re
import zipfile
from pathlib import Path

from lxml import etree
from PIL import Image

from content import CHAPTERS, Bullets, Fig, H3, Numbered, P, Table

TEMPLATE = Path(r"E:\TM471-B-Report-Template.docx")
OUT = Path(__file__).parent / "Report2 Mario Fayez Sobhy 23510186.docx"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
FIGURES = Path(__file__).parent / "figures"
HEADING_STYLES = ("H1", "H2", "H3", "H44")
PAGE_WIDTH_EMU = 5400000                                   # the template's figure width (15 cm)
ARABIC = re.compile(r"[\u0600-\u06FF]")

FACULTY = "Faculty of Artificial Intelligence"
ABBREVIATIONS = [
    ("AI", "Artificial Intelligence"),
    ("BERT", "Bidirectional Encoder Representations from Transformers"),
    ("BMI", "Body Mass Index"),
    ("BMR", "Basal Metabolic Rate"),
    ("CSV", "Comma-Separated Values"),
    ("DASH", "Dietary Approaches to Stop Hypertension"),
    ("DFD", "Data Flow Diagram"),
    ("ERD", "Entity-Relationship Diagram"),
    ("HMAC", "Hash-based Message Authentication Code"),
    ("KNN", "K-Nearest Neighbors"),
    ("ML", "Machine Learning"),
    ("NER", "Named Entity Recognition"),
    ("NLP", "Natural Language Processing"),
    ("PBKDF2", "Password-Based Key Derivation Function 2"),
    ("PDF", "Portable Document Format"),
    ("PIN", "Personal Identification Number"),
    ("SQL", "Structured Query Language"),
    ("SR Legacy", "Standard Reference Legacy (USDA food data release)"),
    ("TDEE", "Total Daily Energy Expenditure"),
    ("UI", "User Interface"),
    ("USDA", "United States Department of Agriculture"),
]
DECLARATION = ("I hereby certify that this material, which I now submit for assessment on the program of study leading to "
               "the award of bachelor of Information Technology and Computing in Computer Science is entirely my own work, "
               "that I have exercised reasonable care to ensure that the work is original, and does not to the best of my "
               "knowledge breach any law of copyright, and has not been taken from the work of others save and to the extent "
               "that such work has been cited and acknowledged within the text of my work.")
ACKNOWLEDGMENTS = [
    "First and foremost, I would like to thank ALLAH for granting me the strength, patience, and ability to complete this "
    "project successfully.",
    "I would like to express my sincere gratitude to my supervisor, Dr. Eid Emary, for his valuable guidance, continuous "
    "support, and constructive feedback throughout all stages of this project. His advice and encouragement were essential "
    "in helping me complete this work.",
    f"I would also like to thank the {FACULTY} at the Arab Open University for providing the academic environment and "
    "resources necessary for completing this project.",
    "Finally, I would like to express my deepest appreciation to my family for their unconditional support, encouragement, "
    "and patience throughout my academic journey.",
]

# Checked on 2 November 2026 against Crossref (papers) or the publisher's own page. The final pass removes any entry that
# the chapters end up not citing.
REFERENCES = [
    "Adomavicius, G. and Tuzhilin, A. (2005) ‘Toward the next generation of recommender systems: a survey of the state-of-the-art and possible extensions’, IEEE Transactions on Knowledge and Data Engineering, 17(6), pp. 734–749.",
    "Afshin, A. et al. (2019) ‘Health effects of dietary risks in 195 countries, 1990–2017: a systematic analysis for the Global Burden of Disease Study 2017’, The Lancet, 393(10184), pp. 1958–1972.",
    "Aggarwal, C.C. (2016) Recommender Systems: The Textbook. Cham: Springer.",
    "Appel, L.J. et al. (1997) ‘A clinical trial of the effects of dietary patterns on blood pressure’, New England Journal of Medicine, 336(16), pp. 1117–1124.",
    "Burke, R. (2002) ‘Hybrid recommender systems: survey and experiments’, User Modeling and User-Adapted Interaction, 12(4), pp. 331–370.",
    "Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) ‘BERT: pre-training of deep bidirectional transformers for language understanding’, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics, pp. 4171–4186.",
    "Dizex (no date) FoodBaseBERT-NER. Available at: https://huggingface.co/Dizex/FoodBaseBERT-NER (Accessed: 2 November 2026).",
    "Eat This Much (2026) Automatic Meal Planner. Available at: https://www.eatthismuch.com/ (Accessed: 2 November 2026).",
    "Elsweiler, D. and Harvey, M. (2015) ‘Towards automatic meal plan recommendations for balanced nutrition’, Proceedings of the 9th ACM Conference on Recommender Systems, pp. 313–316.",
    "Frankenfield, D., Roth-Yousey, L. and Compher, C. (2005) ‘Comparison of predictive equations for resting metabolic rate in healthy nonobese and obese adults: a systematic review’, Journal of the American Dietetic Association, 105(5), pp. 775–789.",
    "Freyne, J. and Berkovsky, S. (2010) ‘Intelligent food planning: personalized recipe recommendation’, Proceedings of the 15th International Conference on Intelligent User Interfaces, pp. 321–324.",
    "Hall, K.D. (2008) ‘What is the required energy deficit per unit weight loss?’, International Journal of Obesity, 32(3), pp. 573–576.",
    "Institute of Medicine (2001) Dietary Reference Intakes for Vitamin A, Vitamin K, Arsenic, Boron, Chromium, Copper, Iodine, Iron, Manganese, Molybdenum, Nickel, Silicon, Vanadium, and Zinc. Washington, DC: National Academies Press.",
    "Institute of Medicine (2011) Dietary Reference Intakes for Calcium and Vitamin D. Washington, DC: National Academies Press.",
    "Khamesian, S., Arefeen, A., Carpenter, S.M. and Ghasemzadeh, H. (2025) ‘NutriGen: personalized meal plan generator leveraging large language models to enhance dietary and nutritional adherence’, arXiv preprint, arXiv:2502.20601.",
    "Lops, P., de Gemmis, M. and Semeraro, G. (2011) ‘Content-based recommender systems: state of the art and trends’, in Ricci, F., Rokach, L., Shapira, B. and Kantor, P.B. (eds.) Recommender Systems Handbook. New York: Springer, pp. 73–105.",
    "Mifflin, M.D. et al. (1990) ‘A new predictive equation for resting energy expenditure in healthy individuals’, The American Journal of Clinical Nutrition, 51(2), pp. 241–247.",
    "Min, W., Jiang, S. and Jain, R. (2020) ‘Food recommendation: framework, existing solutions, and challenges’, IEEE Transactions on Multimedia, 22(10), pp. 2659–2671.",
    "MyFitnessPal (2026) MyFitnessPal. Available at: https://www.myfitnesspal.com/ (Accessed: 2 November 2026).",
    "National Academies of Sciences, Engineering, and Medicine (2019) Dietary Reference Intakes for Sodium and Potassium. Washington, DC: National Academies Press.",
    "Niszczota, P. and Rybicka, I. (2023) ‘The credibility of dietary advice formulated by ChatGPT: robo-diets for people with food allergies’, Nutrition, 112, 112076.",
    "Popovski, G., Seljak, B.K. and Eftimov, T. (2019) ‘FoodBase corpus: a new resource of annotated food entities’, Database, 2019, baz121.",
    "Reimers, N. and Gurevych, I. (2019) ‘Sentence-BERT: sentence embeddings using Siamese BERT-networks’, Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, pp. 3980–3990.",
    "Ricci, F., Rokach, L. and Shapira, B. (eds.) (2015) Recommender Systems Handbook. 2nd edn. New York: Springer.",
    "Stark, P.B. and Parker, R.L. (1995) ‘Bounded-variable least-squares: an algorithm and applications’, Computational Statistics, 10, pp. 129–141.",
    "Stigler, G.J. (1945) ‘The cost of subsistence’, Journal of Farm Economics, 27(2), pp. 303–314.",
    "Stojanov, R., Popovski, G., Cenikj, G., Koroušić Seljak, B. and Eftimov, T. (2021) ‘A fine-tuned bidirectional encoder representations from transformers model for food named-entity recognition: algorithm development and validation’, Journal of Medical Internet Research, 23(8), e28229.",
    "Tran, T.N.T., Atas, M., Felfernig, A. and Stettinger, M. (2018) ‘An overview of recommender systems in the healthy food domain’, Journal of Intelligent Information Systems, 50(3), pp. 501–526.",
    "Trattner, C. and Elsweiler, D. (2017) ‘Food recommender systems: important contributions, challenges and future research directions’, arXiv preprint, arXiv:1711.02760.",
    "U.S. Department of Agriculture, Agricultural Research Service (2018) FoodData Central: SR Legacy, April 2018 release. Available at: https://fdc.nal.usda.gov/download-datasets (Accessed: 2 November 2026).",
    "U.S. Department of Agriculture and U.S. Department of Health and Human Services (2020) Dietary Guidelines for Americans, 2020–2025. 9th edn. Washington, DC: U.S. Government.",
    "Virtanen, P. et al. (2020) ‘SciPy 1.0: fundamental algorithms for scientific computing in Python’, Nature Methods, 17(3), pp. 261–272.",
    "World Health Organization (2026) Healthy Diet. Available at: https://www.who.int/news-room/fact-sheets/detail/healthy-diet (Accessed: 2 November 2026).",
    "Yang, L. et al. (2017) ‘Yum-me: a personalized nutrient-based meal recommender system’, ACM Transactions on Information Systems, 36(1), pp. 1–31.",
    "Zhang, S., Yao, L., Sun, A. and Tay, Y. (2019) ‘Deep learning based recommender system: a survey and new perspectives’, ACM Computing Surveys, 52(1), pp. 1–38.",
]


def text_of(el):
    return "".join(t.text or "" for t in el.iter(W + "t"))


def has_section(p):
    return p.find(f"{W}pPr/{W}sectPr") is not None


def set_text(p, text):
    """Replace a paragraph's text, keeping its paragraph settings, bookmarks and the first run's look (no highlight).
    A newline in the text becomes a line break."""
    runs = [r for r in p.iter(W + "r")]
    rpr = None
    for r in runs:
        if r.find(W + "t") is not None and r.find(W + "rPr") is not None:
            rpr = copy.deepcopy(r.find(W + "rPr"))
            break
    for r in runs:
        r.getparent().remove(r)
    if rpr is not None:
        for tag in ("highlight", "color"):
            for h in rpr.findall(W + tag):
                rpr.remove(h)
    run = etree.SubElement(p, W + "r")
    if rpr is not None:
        run.append(rpr)
    for i, line in enumerate(text.split("\n")):
        if i:
            etree.SubElement(run, W + "br")
        t = etree.SubElement(run, W + "t")
        t.text = line
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")


def clear(p):
    """Empty a paragraph, or remove it if nothing else depends on it (a section break must stay)."""
    if has_section(p):
        for r in list(p.iter(W + "r")):
            r.getparent().remove(r)
    else:
        p.getparent().remove(p)


def find_para(body, start, after=None):
    """First top-level paragraph whose text starts with `start` (optionally only after element `after`).
    Lines of the contents pages are skipped: they repeat the headings."""
    seen = after is None
    for el in body:
        if el is after:
            seen = True
            continue
        if seen and el.tag == W + "p" and text_of(el).strip().startswith(start):
            style = el.find(f"{W}pPr/{W}pStyle")
            if style is None or not style.get(W + "val").startswith(("TOC", "TableofFigures")):
                return el
    raise ValueError(f"Not in the template: {start!r}")


def drop_tips(body):
    """The template's Arabic notes (and the dashed line between them) are instructions, not report text."""
    for p in list(body.iter(W + "p")):
        t = text_of(p).strip()
        if t == "الملخص العربي":
            continue
        if ARABIC.search(t) or re.fullmatch(r"-{10,}", t):
            clear(p)


def cover(body):
    set_text(find_para(body, "Faculty of"), FACULTY)
    set_text(find_para(body, "<Project TITLE>"), "AI Personalized Nutrition")
    set_text(find_para(body, "Subtitle"), "Recommendation System")
    set_text(find_para(body, "Name and ID"), "Mario Fayez Sobhy Bebawe\n23510186")
    last = find_para(body, "TM471B")
    set_text(last, "TM471B: Final Year Project, Fall 2026/2027\nSupervisor: Dr. Eid Emary")
    for el in body:                                        # the template marks its placeholders in yellow
        for h in list(el.iter(W + "highlight")):
            h.getparent().remove(h)
        if el is last:
            break
    for _ in range(3):                                     # the name now takes two lines: keep the cover on one page
        body.remove(last.getnext())


def front_matter(body):
    head = find_para(body, "Declaration (Optional)")
    set_text(head, "Declaration")
    set_text(find_para(body, "We hereby certify"), DECLARATION)

    head = find_para(body, "Acknowledgments (Optional)")
    set_text(head, "Acknowledgments")
    first = find_para(body, "Text here.", head)
    name = find_para(body, "Your Name", head)
    el = first.getnext()
    while el is not name:                                  # empty lines between the text and the name
        following = el.getnext()
        clear(el)
        el = following
    set_text(first, ACKNOWLEDGMENTS[0])
    previous = first
    for line in ACKNOWLEDGMENTS[1:]:
        p = copy.deepcopy(first)
        set_text(p, line)
        previous.addnext(p)
        previous = p
    set_text(name, "Mario Fayez")

    set_text(find_para(body, "List of Figures (Optional)"), "List of Figures")
    set_text(find_para(body, "List of Tables (Optional)"), "List of Tables")
    head = find_para(body, "Abbreviations (Optional)")
    set_text(head, "Abbreviations")
    table = head.getnext()
    while table.tag != W + "tbl":
        table = table.getnext()
    rows = table.findall(W + "tr")
    model = copy.deepcopy(rows[0])
    for r in rows:
        table.remove(r)
    for short, full in ABBREVIATIONS:
        row = copy.deepcopy(model)
        for cell, value in zip(row.findall(W + "tc"), (short, full)):
            paras = cell.findall(W + "p")
            set_text(paras[0], value)
            for extra in paras[1:]:
                cell.remove(extra)
        table.append(row)


def references(body):
    """Replace the template's sample list and notes with the checked Harvard list. Keeps the section breaks."""
    head = next(p for p in body if p.tag == W + "p" and text_of(p).strip() == "References"
                and p.find(f"{W}pPr/{W}pStyle").get(W + "val") == "H1")
    model = None
    el = head.getnext()
    while el is not None and text_of(el).strip() != "الملخص العربي":
        following = el.getnext()
        style = el.find(f"{W}pPr/{W}pStyle") if el.tag == W + "p" else None
        if style is not None and style.get(W + "val") == "EndNoteBibliography":
            if model is None and text_of(el).strip().startswith("Goodfellow"):
                model = copy.deepcopy(el)
            clear(el)
        el = following
    previous = head
    for line in REFERENCES:
        p = copy.deepcopy(model)
        set_text(p, line)
        previous.addnext(p)
        previous = p


# --- chapters -------------------------------------------------------------------------------------------------
class Writer:
    """Turns the blocks of content.py into the template's own paragraph, list, figure and table formats."""

    def __init__(self, body, figure_model):
        self.body, self.figure_model = body, figure_model
        self.images, self.extra_lists = [], []
        self.next_id = 5000

    def style_of(self, el):
        style = el.find(f"{W}pPr/{W}pStyle") if el.tag == W + "p" else None
        return style.get(W + "val") if style is not None else ""

    def heading(self, start):
        for el in self.body:
            if self.style_of(el) in HEADING_STYLES and re.sub(r"\s+", " ", text_of(el)).strip().startswith(start):
                return el
        raise ValueError(f"No heading in the template: {start!r}")

    def paragraph(self, text, style="ALLtext", first_line=None, number=None, keep_next=False, size=None, center=False):
        p = etree.Element(W + "p")
        ppr = etree.SubElement(p, W + "pPr")
        etree.SubElement(ppr, W + "pStyle").set(W + "val", style)
        if keep_next:
            etree.SubElement(ppr, W + "keepNext")
        if number is not None:
            num = etree.SubElement(ppr, W + "numPr")
            etree.SubElement(num, W + "ilvl").set(W + "val", "0")
            etree.SubElement(num, W + "numId").set(W + "val", str(number))
        spacing = etree.SubElement(ppr, W + "spacing")
        spacing.set(W + "line", "276")
        spacing.set(W + "lineRule", "auto")
        if first_line is not None:
            etree.SubElement(ppr, W + "ind").set(W + "firstLine", str(first_line))
        if center:
            etree.SubElement(ppr, W + "jc").set(W + "val", "center")
        for i, part in enumerate(re.split(r"\*\*", text)):               # **bold**
            if not part:
                continue
            run = etree.SubElement(p, W + "r")
            rpr = etree.SubElement(run, W + "rPr")
            if i % 2:
                etree.SubElement(rpr, W + "b")
            color = etree.SubElement(rpr, W + "color")
            color.set(W + "val", "000000")
            if size:
                etree.SubElement(rpr, W + "sz").set(W + "val", str(size))
                etree.SubElement(rpr, W + "szCs").set(W + "val", str(size))
            t = etree.SubElement(run, W + "t")
            t.text = part
            t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        return p

    def subheading(self, text, model):
        p = copy.deepcopy(model)
        for tag in ("bookmarkStart", "bookmarkEnd"):
            for b in p.findall(W + tag):
                p.remove(b)
        set_text(p, text)
        return p

    def figure(self, block):
        table = copy.deepcopy(self.figure_model)
        with Image.open(FIGURES / block.file) as im:
            ratio = im.height / im.width
        cx = int(PAGE_WIDTH_EMU * block.width)
        cy = int(cx * ratio)
        rel = f"rIdFig{len(self.images) + 1}"
        self.images.append((rel, block.file))
        for ext in list(table.iter(WP + "extent")) + list(table.iter(A + "ext")):
            if ext.get("cx"):
                ext.set("cx", str(cx))
                ext.set("cy", str(cy))
        for doc_pr in table.iter(WP + "docPr"):
            self.next_id += 1
            doc_pr.set("id", str(self.next_id))
            doc_pr.set("name", f"Figure {self.next_id}")
            doc_pr.set("descr", block.caption)
        for blip in table.iter(A + "blip"):
            blip.set(f"{{{R_NS}}}embed", rel)
        rows = table.findall(W + "tr")
        for row in rows:
            height = row.find(f"{W}trPr/{W}trHeight")
            if height is not None:
                height.getparent().remove(height)
            etree.SubElement(row.find(W + "trPr"), W + "cantSplit")
        image_cell_para = rows[0].find(f"{W}tc/{W}p")
        etree.SubElement(image_cell_para.find(W + "pPr"), W + "keepNext")
        caption = rows[1].find(W + "tc").findall(W + "p")
        set_text(caption[0], f"{block.number} {block.caption}")
        for extra in caption[1:]:
            rows[1].find(W + "tc").remove(extra)
        for tag in ("bookmarkStart", "bookmarkEnd"):
            for b in caption[0].findall(W + tag):
                caption[0].remove(b)
        return table

    def table(self, block):
        caption = self.paragraph(f"**{block.number} {block.title}**", style="TableCaption0", keep_next=True, size=20)
        total = 8400
        widths = [int(total * w / sum(block.widths)) for w in block.widths]
        tbl = etree.Element(W + "tbl")
        pr = etree.SubElement(tbl, W + "tblPr")
        tw = etree.SubElement(pr, W + "tblW")
        tw.set(W + "w", str(sum(widths)))
        tw.set(W + "type", "dxa")
        etree.SubElement(pr, W + "jc").set(W + "val", "center")
        etree.SubElement(pr, W + "tblLayout").set(W + "type", "fixed")
        margins = etree.SubElement(pr, W + "tblCellMar")
        for side in ("left", "right"):
            m = etree.SubElement(margins, W + side)
            m.set(W + "w", "60")
            m.set(W + "type", "dxa")
        grid = etree.SubElement(tbl, W + "tblGrid")
        for w in widths:
            etree.SubElement(grid, W + "gridCol").set(W + "w", str(w))
        rows = [block.header] + block.rows
        for r, values in enumerate(rows):
            row = etree.SubElement(tbl, W + "tr")
            trpr = etree.SubElement(row, W + "trPr")
            etree.SubElement(trpr, W + "cantSplit")
            if r == 0:
                etree.SubElement(trpr, W + "tblHeader")
            for w, value in zip(widths, values):
                cell = etree.SubElement(row, W + "tc")
                tcpr = etree.SubElement(cell, W + "tcPr")
                tcw = etree.SubElement(tcpr, W + "tcW")
                tcw.set(W + "w", str(w))
                tcw.set(W + "type", "dxa")
                borders = etree.SubElement(tcpr, W + "tcBorders")
                for side, val, on in (("top", "double", r == 0), ("bottom", "single", r == 0), ("bottom", "double", r == len(rows) - 1)):
                    if on and borders.find(W + side) is None:
                        b = etree.SubElement(borders, W + side)
                        b.set(W + "val", val)
                        b.set(W + "sz", "4")
                        b.set(W + "space", "0")
                        b.set(W + "color", "auto")
                text = str(value)
                para = self.paragraph(f"**{text}**" if r == 0 else text, style="Normal", size=block.size)
                ppr = para.find(W + "pPr")
                ppr.find(W + "spacing").set(W + "line", "240")
                ppr.find(W + "spacing").set(W + "before", "30")
                ppr.find(W + "spacing").set(W + "after", "30")
                etree.SubElement(ppr, W + "jc").set(W + "val", "left")
                cell.append(para)
        return [caption, tbl, self.paragraph("", first_line=0)]

    def render(self, block, models):
        if isinstance(block, P):
            return [self.paragraph(block.text)]
        if isinstance(block, Bullets):
            return [self.paragraph(x, number=18, first_line=0) for x in block.items]
        if isinstance(block, Numbered):
            self.extra_lists.append(f"{100 + len(self.extra_lists)}")
            return [self.paragraph(x, number=self.extra_lists[-1], first_line=0) for x in block.items]
        if isinstance(block, H3):
            return [self.subheading(block.text, models["H3"])]
        if isinstance(block, Fig):
            return [self.figure(block), self.paragraph("", first_line=0)]
        if isinstance(block, Table):
            return self.table(block)
        raise TypeError(block)

    def fill(self, start, blocks, models, rename=None):
        """Put blocks under a heading. Everything of the template up to the next heading of the same level or higher goes;
        section breaks and page breaks stay, because the headers and page numbers depend on them."""
        head = self.heading(start)
        level = HEADING_STYLES.index(self.style_of(head))
        el = head.getnext()
        while el is not None:
            style = self.style_of(el)
            if style in HEADING_STYLES and HEADING_STYLES.index(style) <= level:
                break
            following = el.getnext()
            if el.tag == W + "tbl":
                self.body.remove(el)
            elif el.tag == W + "p":
                if has_section(el):
                    clear(el)
                elif any(b.get(W + "type") == "page" for b in el.iter(W + "br")):
                    pass
                else:
                    self.body.remove(el)
            el = following
        if rename:
            set_text(head, rename)
        previous = head
        for block in blocks:
            for new in self.render(block, models):
                previous.addnext(new)
                previous = new


def write_chapters(body):
    figure_model = next(el for el in body if el.tag == W + "tbl" and text_of(el).startswith("Figure 1-1"))
    figure_model = copy.deepcopy(figure_model)
    writer = Writer(body, figure_model)
    models = {"H3": copy.deepcopy(writer.heading("2.1.1 Subsection"))}
    for section in CHAPTERS:
        writer.fill(section["heading"], section["blocks"], models, section.get("rename"))
    return writer


def add_lists(numbering_xml, ids):
    """Each numbered list starts again at 1."""
    root = etree.fromstring(numbering_xml)
    abstract = next(n.find(W + "abstractNumId").get(W + "val") for n in root.findall(W + "num") if n.get(W + "numId") == "19")
    for num_id in ids:
        num = etree.SubElement(root, W + "num")
        num.set(W + "numId", num_id)
        etree.SubElement(num, W + "abstractNumId").set(W + "val", abstract)
        over = etree.SubElement(num, W + "lvlOverride")
        over.set(W + "ilvl", "0")
        etree.SubElement(over, W + "startOverride").set(W + "val", "1")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def add_images(rels_xml, images):
    root = etree.fromstring(rels_xml)
    for rel, file in images:
        r = etree.SubElement(root, "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
        r.set("Id", rel)
        r.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
        r.set("Target", f"media/{file}")
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def fix_header(xml):
    """Headers carry the faculty's name, and one has a typo."""
    return xml.replace(b"Faculty of Computer Studies", FACULTY.encode()).replace(b"Refernces", b"References")


def build():
    src = zipfile.ZipFile(TEMPLATE)
    root = etree.fromstring(src.read("word/document.xml"))
    body = root.find(W + "body")
    drop_tips(body)
    cover(body)
    front_matter(body)
    references(body)
    writer = write_chapters(body)
    document = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as out:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/document.xml":
                data = document
            elif item.filename == "word/numbering.xml":
                data = add_lists(data, writer.extra_lists)
            elif item.filename == "word/_rels/document.xml.rels":
                data = add_images(data, writer.images)
            elif re.fullmatch(r"word/header\d+\.xml", item.filename):
                data = fix_header(data)
            out.writestr(item, data)
        for file in {f for _, f in writer.images}:
            out.write(FIGURES / file, f"word/media/{file}")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    build()
