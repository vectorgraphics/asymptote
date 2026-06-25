#!/usr/bin/env python3
"""Python port of bfnnconv.pl + m-ascii.pl/m-info.pl/m-html.pl.

Reads a .bfnn file (the "Bizarre Format With No Name") and produces the
ASCII, Info and HTML outputs, byte-for-byte compatible with the original
Perl toolchain by Ian Jackson.  Like the Perl version, it relies on a
two-pass build: it reads <prefix>.xrefdb (from the previous run) to resolve
cross references and writes an updated one, so it must be run twice.

Bytes are processed as latin-1 so that len()/regex/indexing match Perl's
byte semantics exactly.
"""

# This module is a single-file, faithful port of a Perl toolchain.  It keeps
# the Perl program's shared document state as module globals (the block below),
# so the `global` statements that maintain that state are intentional, as is
# the module length needed to hold all four output backends together.
# pylint: disable=too-many-lines,global-statement

import os
import re
import shutil
import sys
import time

A = re.ASCII

# ---------------------------------------------------------------------------
# shared state (mirrors the Perl globals)
# ---------------------------------------------------------------------------
qrefn = {}  # ref -> "s.q"
qreft = {}  # ref -> question text
qn2ref = {}  # (s,n) -> ref
sn2title = {}  # section number -> title
maxsection = 0
maxquestion = {}  # section -> max question number
user = {}  # \set variables
holdover = ""
styles = []  # @styles stack
U = []  # xrefdb-new lines
outputs = []  # list of active backend objects
section = -1
question = 0
label = ""  # the stray global $label referenced in info_endindexitem
prefix = ""  # output basename, set in main()
backends = {}  # name -> backend instance, populated in main()


def epoch():
    return int(os.environ.get("SOURCE_DATE_EPOCH") or time.time())


def strftime_utc(fmt):
    return time.strftime(fmt, time.gmtime(epoch()))


# ---------------------------------------------------------------------------
# backend protocol
#
# The driver dispatches each BFNN markup verb to a like-named method on every
# active backend, so each backend must answer to every verb.  The backend
# classes below therefore expose one (often trivial) method per verb; their
# wide method and attribute counts are inherent to that protocol rather than a
# sign of a class doing too much.
# ---------------------------------------------------------------------------
class Backend:
    """Defaults shared by every backend.

    Markup verbs a backend does not render fall through to the no-ops here;
    backends override only the ones they care about.  ``arg``/``endarg``
    provide the argument buffering used by verbs that collect their body text
    before emitting it (only HTML and Texinfo use it).
    """

    def __init__(self):
        self.cmds = []  # stack of buffering markup verbs
        self.args = []  # matching stack of collected argument text

    def ftpon(self):
        pass

    def endftpon(self):
        pass

    def ftpin(self):
        pass

    def endftpin(self):
        pass

    def docref(self):
        pass

    def enddocref(self):
        pass

    def courier(self):
        pass

    def endcourier(self):
        pass

    def newsgroup(self):
        pass

    def endnewsgroup(self):
        pass

    def finish(self):
        pass

    def arg(self, cmd):
        self.cmds.append(cmd)
        self.args.append("")

    def endarg(self):
        cmd = self.cmds.pop()
        body = self.args.pop()
        getattr(self, "do_" + cmd)(body)


class WrappingBackend(Backend):
    """Shared machinery for the line-wrapping plain-text backends (ASCII, Info).

    Both collect output into ``self.buf`` and reflow running text held in
    ``self.para`` through :meth:`writepara` before flushing it.
    """

    # pylint: disable=too-many-instance-attributes,too-many-public-methods

    def __init__(self):
        super().__init__()
        self.buf = []
        self.status = ""
        self.para = ""
        self.ignore = 0
        self.indentstring = ""
        self.nextindent = ""
        self.plc = 0

    def w(self, s):
        self.buf.append(s)

    def italic(self):
        self.text("*")

    def enditalic(self):
        self.para += "*"

    def email(self):
        self.text("<")

    def endemail(self):
        self.text(">")

    def ftpsilent(self):
        self.ignore += 1

    def endftpsilent(self):
        self.ignore -= 1

    def text(self, s):
        if self.ignore:
            return
        if self.status == "":
            self.status = "p"
        self.para += s

    def tab(self, n):
        n = int(n) - len(self.para)
        if n > 0:
            self.para += " " * n

    def writepara(self):
        output = ""
        while re.search(r"\S", self.para):
            thisline = self.indentstring
            while True:
                m = re.match(r"(\s*\S+)", self.para)
                if not m:
                    break
                word = m.group(1)
                fits = len(word) + len(thisline) < 75
                if not (fits or len(thisline) == len(self.indentstring)):
                    break
                thisline += word
                self.para = self.para[m.end() :]
            self.para = re.sub(r"^\s*", "", self.para)
            output += thisline + "\n"
            self.indentstring = self.nextindent
            if not self.para:
                break
        self.status = ""
        self.para = ""
        return output

    def newline(self):
        if self.status != "p":
            return
        self.w(self.writepara())

    def endpara(self):
        if self.status != "p":
            return
        self.w(self.writepara())
        self.w("\n")

    def endheading(self):
        self.para = re.sub(r"\s*$", "", self.para)
        self.w(f"{self.para}\n\n")
        self.status = ""
        self.para = ""

    def endmajorheading(self, *_):
        self.endheading()

    def endminorheading(self, *_):
        self.endheading()

    def verbatim(self, s):
        self.w(s + "\n")

    def item(self):
        self.newline()
        self.indentstring = re.sub(r"  $", "* ", self.indentstring)

    def endpageref(self):
        self.text("'")

    def startpackedlist(self):
        self.plc = 0

    def endpackedlist(self):
        if not self.plc:
            self.newline()

    def packeditem(self):
        if not self.plc:
            self.newline()
        self.tab(self.plc * 40 + 5)
        self.plc = 0 if self.plc else 1


# ---------------------------------------------------------------------------
# ASCII backend (m-ascii.pl)
# ---------------------------------------------------------------------------
class Ascii(WrappingBackend):
    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    name = "ascii"

    def __init__(self):
        super().__init__()
        self.left = ""
        self.vstatus = ""
        self.istatus = ""

    def init(self):
        pass  # file opened lazily; output collected in self.buf

    def startmajorheading(self, *a):
        self.w("=" * 79 + "\n\n")
        self.status = "h"
        self.text(f"Section {a[0]}.  " if truthy(a[0]) else "")

    def startminorheading(self, *_):
        self.w("-" * 79 + "\n\n")
        self.status = "h"

    def startverbatim(self):
        self.vstatus = self.status
        self.w(self.writepara())

    def endverbatim(self):
        self.status = self.vstatus

    def write_files(self):
        with open(prefix + ".ascii", "w", encoding="latin-1") as f:
            f.write("".join(self.buf))

    def startindex(self):
        self.status = ""

    def endindex(self):
        self.status = "p"

    def endindexitem(self):
        self.w(f" {self.left:<11} {self.para:.66}\n")
        self.status = "p"
        self.para = ""

    def startindexitem(self, *a):
        self.left = a[1]

    def startindexmainitem(self, *a):
        self.left = a[1]
        if self.status == "p":
            self.w("\n")

    def startindent(self):
        self.istatus = self.status
        self.w(self.writepara())
        self.indentstring = "   " + self.indentstring
        self.nextindent = "   " + self.nextindent

    def endindent(self):
        self.indentstring = re.sub(r"^   ", "", self.indentstring)
        self.nextindent = re.sub(r"^   ", "", self.nextindent)
        self.status = self.istatus

    def startlist(self):
        self.endpara()
        self.indentstring = "  " + self.indentstring
        self.nextindent = "  " + self.nextindent

    def endlist(self):
        self.endpara()
        self.indentstring = re.sub(r"^  ", "", self.indentstring)
        self.nextindent = re.sub(r"^  ", "", self.nextindent)

    def pageref(self, *a):
        self.text(f"Q{a[1]} `")


# ---------------------------------------------------------------------------
# Info backend (m-info.pl)
# ---------------------------------------------------------------------------
class Info(WrappingBackend):
    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    name = "info"

    def __init__(self):
        super().__init__()
        self.moredetail = ""
        self.main = 0
        self.lab = ""
        self.istatus = ""

    def init(self):
        self.w(
            f"Info file: {prefix}.info,    -*-Text-*-\n"
            "produced by bfnnconv.py from the Bizarre Format With No Name.\n\n"
        )

    def heading(self, *a):
        # refstring node next previous up
        # The literal 0x1F is the GNU Info node-separator byte (see m-info.pl).
        self.w(f"\x1f\nFile: {prefix}.info, Node: {a[1]}")
        if len(a) > 2 and a[2]:
            self.w(f", Next: {a[2]}")
        if len(a) > 3 and a[3]:
            self.w(f", Previous: {a[3]}")
        if len(a) > 4 and a[4]:
            self.w(f", Up: {a[4]}")
        self.w("\n\n")
        self.status = ""

    def startmajorheading(self, *a):
        if a[0] == "0":
            return
        self.heading("s_" + a[0], *(list(a[1:]) + ["Top"]))

    def startminorheading(self, *a):
        self.heading(*a)

    def startverbatim(self):
        self.w(self.writepara())

    def endverbatim(self):
        self.status = ""

    def write_files(self):
        with open(prefix + ".info", "w", encoding="latin-1") as f:
            f.write("".join(self.buf))

    def startindex(self):
        self.endpara()
        self.moredetail = ""
        self.status = ""

    def endindex(self):
        if self.moredetail:
            self.w(f"{self.moredetail}\n")

    def endindexitem(self):
        self.indentstring = f"* {self.lab + '::':<17} "
        self.nextindent = " " * 20
        txt = self.writepara()
        if self.main:
            self.w(label + txt)
            txt = re.sub(r"^.{20}", "", txt, flags=re.S)
            self.moredetail += txt
        else:
            self.moredetail += label + txt
        self.indentstring = self.nextindent = ""
        self.status = "p"

    def startindexitem(self, *a):
        if self.status == "":
            self.w("* Menu:\n")
        self.status = ""
        self.lab = a[2]
        self.main = 0

    def startindexmainitem(self, *a):
        if self.status == "":
            self.w("* Menu:\n")
        self.lab = a[2]
        self.main = 1
        self.moredetail += f"\n{a[2]}, "
        self.status = ""

    def startindent(self):
        self.istatus = self.status
        self.w(self.writepara())
        self.indentstring = "   " + self.indentstring
        self.nextindent = "   " + self.nextindent

    def endindent(self):
        self.indentstring = re.sub(r"^   ", "", self.indentstring)
        self.nextindent = re.sub(r"^   ", "", self.nextindent)
        self.status = self.istatus

    def startlist(self):
        self.istatus = self.status
        self.w(self.writepara())
        self.indentstring = "  " + self.indentstring
        self.nextindent = "  " + self.nextindent

    def endlist(self):
        self.indentstring = re.sub(r"^  ", "", self.indentstring)
        self.nextindent = re.sub(r"^  ", "", self.nextindent)
        self.status = ""

    def pageref(self, *a):
        self.text(f"*Note Question {a[1]}:: `")


# ---------------------------------------------------------------------------
# HTML backend (m-html.pl)
# ---------------------------------------------------------------------------
SANI = {"<": "lt", ">": "gt", "&": "amp", '"': "quot"}


class Html(Backend):
    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    name = "html"

    def __init__(self):
        super().__init__()
        self.files = {}  # filename -> list of strings
        self.cur = None
        self.needpara = 0
        self.end = ""
        self.sectionn = ""
        self.date = ""
        self.year = ""
        self.copyrighthref = ""
        self.refval = {}
        self.indexunhead = ""
        self.itemend = ""
        self.tabignore = 0
        self.ftpsite = ""
        self.ftpdir = ""

    def w(self, s):
        self.cur.append(s)

    def open(self, fname):
        self.files[fname] = []
        self.cur = self.files[fname]

    def init(self):
        self.open("index.html")
        self.w('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2//EN">\n')
        self.w("<html>\n")
        self.needpara = -1
        self.end = ""
        self.date = strftime_utc("%Y-%m-%d")
        self.year = strftime_utc("%Y")

    def close(self):
        self.w(self.end + f"<address>\n{user.get('author', '')}\n")
        self.w(f"- {self.date}\n</address><br>\n")
        self.w(f"Extracted from {user.get('title', '')},\n")
        if self.copyrighthref:
            self.w(f'<A href="{self.copyrighthref}">')
        self.w(f"Copyright &copy; {self.year} {user.get('copyholder', '')}.")
        if self.copyrighthref:
            self.w("</A>")
        self.w("\n</body></html>\n")

    def startmajorheading(self, *a):
        ref, this, nxt, back = a
        this = re.sub(r"^Section ", "section", this)
        self.sectionn = ref
        nextt = ""
        m = re.match(r"^Section (.*)", nxt, A)
        if m:
            nxt = "section" + m.group(1)
            nextt = sn2title.get(_int_or(m.group(1)), "")
        backt = ""
        m = re.match(r"^Section (.*)", back, A)
        if m:
            back = "section" + m.group(1)
            backt = sn2title.get(_int_or(m.group(1)), "")
        else:
            back = ""
        if truthy(self.sectionn):
            self.close()
            self.open(this + ".html")
            self.w('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 3.2//EN">\n')
            self.w("<html>\n")
            self.end = "<hr>\n"
            if nxt:
                self.end += (
                    f'Next: <a href="{nxt}.html" rel=precedes>{nextt}</a>.<br>\n'
                )
            if back:
                self.end += (
                    f'Back: <a href="{back}.html" rev=precedes>{backt}</a>.<br>\n'
                )
            self.end += '<a href="index.html" rev=subdocument>'
            self.end += "Return to contents</a>.<p>\n"
            self.w(
                f"<head><title>\n{user.get('brieftitle', '')}"
                f" - Section {self.sectionn}\n</title>\n"
            )
            self.w(f'<link rev="made" href="mailto:{user.get("authormail", "")}">\n')
            self.w('<link rel="Contents" href="index.html">\n')
            self.w('<link rel="Start" href="index.html">\n')
            if nxt:
                self.w(f'<link rel="Next" href="{nxt}.html">')
            if back:
                self.w(f'<link rel="Previous" href="{back}.html">')
            self.w('<link rel="Bookmark" title="Asymptote FAQ" href="index.html">\n')
            self.w('</head><body text="#000000" bgcolor="#FFFFFF"><h1>\n')
            self.w(f"{user.get('brieftitle', '')} - Section {self.sectionn} <br>\n")
            self.needpara = -1
        else:
            self.w("\n<h1>\n")
            self.needpara = -1

    def endmajorheading(self, *_):
        self.w("\n</h1>\n\n")
        self.needpara = -1

    def startminorheading(self, *a):
        this = a[1]
        self.needpara = 0
        m = re.match(r"^Question (\d+)\.(\d+)", this, A)
        s, n = m.group(1), m.group(2)
        self.w(f'\n<h2><A name="{qn2ref.get((s, n), "")}">\n')

    def endminorheading(self, *_):
        self.w("\n</A></h2>\n\n")
        self.needpara = -1

    def newsgroup(self):
        self.arg("newsgroup")

    def endnewsgroup(self):
        self.endarg()

    def do_newsgroup(self, a):
        self.w(f'<A href="news:{a}"><code>{a}</code></A>')

    def email(self):
        self.arg("email")

    def endemail(self):
        self.endarg()

    def do_email(self, a):
        self.w(f'<A href="mailto:{a}"><code>{a}</code></A>')

    def courier(self):
        self.w("<code>")

    def endcourier(self):
        self.w("</code>")

    def italic(self):
        self.w("<i>")

    def enditalic(self):
        self.w("</i>")

    def docref(self):
        self.arg("docref")

    def enddocref(self):
        self.endarg()

    def do_docref(self, a):
        if a not in self.refval:
            self.w(f'<A href="{a}">')
        else:
            self.w(f'<A href="{self.refval[a]}">')
        recurse(self, a)
        self.w("</A>")

    def ftpsilent(self):
        self.arg("ftpsilent")

    def endftpsilent(self):
        self.endarg()

    def do_ftpsilent(self, a):
        if ":" in a:
            i = a.index(":")
            self.ftpsite = a[:i]
            self.ftpdir = a[i + 1 :] + "/"
        else:
            self.ftpsite = a
            self.ftpdir = ""

    def ftpon(self):
        self.arg("ftpon")

    def endftpon(self):
        self.endarg()

    def do_ftpon(self, a):
        self.ftpsite = a
        self.ftpdir = ""
        self.w("<code>")
        recurse(self, a)
        self.w("</code>")

    def ftpin(self):
        self.arg("ftpin")

    def endftpin(self):
        self.endarg()

    def do_ftpin(self, a):
        self.w(f'<A href="ftp://{self.ftpsite}{self.ftpdir}{a}"><code>')
        recurse(self, a)
        self.w("</code></A>")

    def text(self, s):
        if self.needpara > 0:
            self.w("\n<p>\n")
        self.needpara = 0
        stuff = sanitise(s)
        while True:
            m = re.match(r"(.{40,70}) ", stuff)
            if not m:
                break
            self.w(f"{m.group(1)}\n")
            stuff = stuff[m.end() :]
        self.w(stuff)

    def tab(self, *_):
        if self.tabignore:
            sys.stderr.write("html tab ignored\n")
        self.tabignore += 1

    def newline(self):
        self.w("<br>\n")

    def startverbatim(self):
        self.w("<pre>\n")

    def verbatim(self, s):
        self.w(sanitise(s) + "\n")

    def endverbatim(self):
        self.w("</pre>\n")
        self.needpara = -1

    def endpara(self):
        if not self.needpara:
            self.needpara += 1

    def finish(self):
        self.close()

    def write_files(self):
        htmldir = "./" + prefix + ".html"
        htmldir = re.sub(r"^\.//", "/", htmldir)
        if os.path.isdir(htmldir):
            shutil.rmtree(htmldir)
        os.makedirs(htmldir)
        for fname, chunks in self.files.items():
            with open(os.path.join(htmldir, fname), "w", encoding="latin-1") as f:
                f.write("".join(chunks))

    def startindex(self):
        self.w("<ul>\n")

    def endindex(self):
        self.w("</ul><hr>\n")

    def startindexitem(self, *a):
        qval = a[1]
        m = re.match(r"Q(\d+)\.(\d+)", qval, A)
        s, n = m.group(1), m.group(2)
        self.w('<li><a href="')
        self.w("" if int(s) == _int_or(self.sectionn) else f"section{s}.html")
        self.w(f'#{qn2ref.get((s, n), "")}" rel=subdocument>Q{s}.{n}. ')
        self.indexunhead = ""

    def startindexmainitem(self, *a):
        s = re.search(r"\d+", a[1]).group(0)
        if int(s) > 1:
            self.w("<br><br>")
        self.w(
            f'<li><b><font size="+2"><a href="section{s}.html" '
            f"rel=subdocument>Section {s}.  "
        )
        self.indexunhead = "</font></b>"

    def endindexitem(self):
        self.w(f"</a>{self.indexunhead}\n")

    def startlist(self):
        self.w("\n")
        self.itemend = "<ul>"

    def endlist(self):
        self.w(f"{self.itemend}\n</ul>\n")
        self.needpara = -1

    def item(self):
        self.w(f"{self.itemend}\n<li>")
        self.itemend = ""
        self.needpara = -1

    def startpackedlist(self):
        self.w("\n")
        self.itemend = "<dir>"

    def endpackedlist(self):
        self.w(f"{self.itemend}\n</dir>\n")
        self.needpara = -1

    def packeditem(self):
        self.w(f"{self.itemend}\n<li>")
        self.itemend = ""
        self.needpara = -1

    def startindent(self):
        self.w("<blockquote>\n")

    def endindent(self):
        self.w("</blockquote>\n")

    def pageref(self, *a):
        sq = a[1]
        m = re.match(r"(\d+)\.(\d+)", sq, A)
        s, n = m.group(1), m.group(2)
        self.w('<A href="')
        self.w("" if int(s) == _int_or(self.sectionn) else f"section{s}.html")
        self.w(f'#{qn2ref.get((s, n), "")}">Q{sq} `')

    def endpageref(self):
        self.w("'</A>")


# ---------------------------------------------------------------------------
# Texinfo backend (no Perl original; see bfnnconv.py.NOTES.md)
#
# This builds an in-memory model of the document (sections + questions) and
# emits a single asy-faq.texi at write time.  makeinfo then produces the
# info / html / plaintext / pdf outputs from that .texi.  The structure is
# chosen to *resemble* the original BFNN outputs; it is intentionally not
# byte-identical (texinfo owns numbering, navigation and wrapping).
# ---------------------------------------------------------------------------
def texesc(s):
    return s.replace("@", "@@").replace("{", "@{").replace("}", "@}")


class Texi(Backend):
    # pylint: disable=too-many-instance-attributes,too-many-public-methods
    name = "texi"

    def __init__(self):
        super().__init__()
        self.top_intro = ""
        self.sections = []  # real chapters (bfnn sections 1..N)
        self.cur = None  # current section dict
        self.q = None  # current question dict
        self.target = "top"  # top|sectitle|secintro|qtitle|qbody|None
        self.in_index = False
        self.swallow = False  # drop the qref display text after @ref
        self.used_nodes = set()
        self.date = ""
        self.code_open = 0  # unbalanced @code{ from source quirks (see NOTES)

    def init(self):
        self.date = strftime_utc("%Y-%m-%d")

    # -- text routing -------------------------------------------------------
    def append_raw(self, s):
        if self.in_index or self.swallow:
            return
        t = self.target
        if t == "top":
            self.top_intro += s
        elif t == "sectitle":
            self.cur["title_texi"] += s
        elif t == "secintro":
            self.cur["intro"] += s
        elif t == "qtitle":
            self.q["title_texi"] += s
        elif t == "qbody":
            self.q["body"] += s

    def append_plain(self, s):
        if self.in_index or self.swallow:
            return
        t = self.target
        if t == "sectitle":
            self.cur["title_plain"] += s
        elif t == "qtitle":
            self.q["title_plain"] += s

    def text(self, s):
        self.append_raw(texesc(s))
        self.append_plain(s)

    def endpara(self):
        if self.target in ("top", "secintro", "qbody"):
            self.append_raw("\n\n")

    def _balance(self):
        # Close any @code{ left open by an unbalanced courier span in the
        # source (the canonical sources are fixed, but stay robust to it).
        while self.code_open > 0:
            self.append_raw("}")
            self.code_open -= 1

    # -- structure ----------------------------------------------------------
    def startmajorheading(self, *a):
        ref = a[0]
        self._balance()
        if not truthy(ref):
            # bfnn section 0 ("Index"): becomes the Top node; ignore its title
            # and its generated \index listing.
            self.cur = None
            self.target = None
            return
        self.cur = {
            "num": int(ref),
            "title_texi": "",
            "title_plain": "",
            "intro": "",
            "questions": [],
        }
        self.sections.append(self.cur)
        self.target = "sectitle"

    def endmajorheading(self, *_):
        if self.cur is not None:
            self.target = "secintro"
        else:
            self.target = None

    def startminorheading(self, *a):
        self._balance()
        anchor = a[0]
        node = anchor
        n = 2
        while node in self.used_nodes:
            node = f"{anchor}_{n}"
            n += 1
        self.used_nodes.add(node)
        self.q = {
            "anchor": anchor,
            "node": node,
            "title_texi": "",
            "title_plain": "",
            "body": "",
        }
        self.cur["questions"].append(self.q)
        self.target = "qtitle"

    def endminorheading(self, *_):
        self._balance()
        self.target = "qbody"

    # the generated index listings are rebuilt from the model at write time
    def startindex(self):
        self.in_index = True

    def endindex(self):
        self.in_index = False

    def startindexitem(self, *_):
        pass

    def startindexmainitem(self, *_):
        pass

    def endindexitem(self):
        pass

    # -- inline styles ------------------------------------------------------
    def courier(self):
        self.append_raw("@code{")
        self.code_open += 1

    def endcourier(self):
        if self.code_open > 0:
            self.append_raw("}")
            self.code_open -= 1

    def italic(self):
        self.append_raw("@emph{")

    def enditalic(self):
        self.append_raw("}")

    def docref(self):
        self.arg("docref")

    def enddocref(self):
        self.endarg()

    def do_docref(self, a):
        self.append_raw("@uref{" + texesc(a) + "}")

    def email(self):
        self.arg("email")

    def endemail(self):
        self.endarg()

    def do_email(self, a):
        self.append_raw("@email{" + texesc(a) + "}")

    def newsgroup(self):
        self.arg("newsgroup")

    def endnewsgroup(self):
        self.endarg()

    def do_newsgroup(self, a):
        self.append_raw("@code{" + texesc(a) + "}")

    def pageref(self, *a):
        # a[0] = anchor, a[1] = "s.q"; emit a cross reference displayed as
        # "Question s.q" and drop the text the driver appends (see endpageref).
        if len(a) > 1 and a[1]:
            self.append_raw("@ref{" + a[0] + ",,Question " + a[1] + "}")
        else:
            self.append_raw("@ref{" + a[0] + "}")
        self.swallow = True

    def endpageref(self):
        self.swallow = False

    def tab(self, *_):
        pass

    def newline(self):
        self.append_raw("@*\n")

    # -- verbatim -----------------------------------------------------------
    def startverbatim(self):
        self.append_raw("\n@verbatim\n")

    def verbatim(self, s):
        self.append_raw(s + "\n")

    def endverbatim(self):
        self.append_raw("@end verbatim\n")

    def finish(self):
        self._balance()

    # -- emit ---------------------------------------------------------------
    def write_files(self):
        title = user.get("title", "")
        o = []
        o.append("\\input texinfo @c -*-texinfo-*-\n")
        o.append(f"@c Generated from {prefix}.bfnn by bfnnconv.py -- do not edit.\n")
        o.append("@c %**start of header\n")
        o.append(f"@setfilename {prefix}.info\n")
        o.append(f"@settitle {title}\n")
        # Source is pure ASCII; US-ASCII keeps makeinfo's output close to the
        # original (straight quotes instead of directed Unicode quotes).
        o.append("@documentencoding US-ASCII\n")
        o.append("@c %**end of header\n\n")
        o.append(f"@copying\n{title}\n\n{self.date}\n@end copying\n\n")
        o.append("@dircategory Languages\n")
        o.append(
            "@direntry\n* asymptote FAQ: (asy-faq).    "
            "Asymptote Frequently Asked Questions.\n@end direntry\n\n"
        )

        o.append(f"@node Top\n@top {title}\n\n")
        intro = re.sub(r"\n{3,}", "\n\n", self.top_intro).strip()
        if intro:
            o.append(intro + "\n\n")
        o.append(f"@center {self.date}\n\n")

        o.append("@menu\n")
        for s in self.sections:
            o.append(f"* {s['title_plain'].strip()}::\n")
        o.append("@end menu\n\n")

        for s in self.sections:
            stitle = s["title_plain"].strip()
            o.append(f"@node {stitle}\n@chapter {s['title_texi'].strip()}\n\n")
            secintro = re.sub(r"\n{3,}", "\n\n", s["intro"]).strip()
            if secintro:
                o.append(secintro + "\n\n")
            o.append("@menu\n")
            for q in s["questions"]:
                desc = texesc(re.sub(r"\s+", " ", q["title_plain"]).strip())
                o.append(f"* {q['node']}:: {desc}\n")
            o.append("@end menu\n\n")
            for q in s["questions"]:
                o.append(f"@node {q['node']}\n")
                # Drop the redundant "Question N.M." prefix: texinfo numbers
                # the @section automatically (the menu keeps the full label).
                stitle = re.sub(r"^Question \d+\.\d+\.\s+", "", q["title_texi"].strip())
                o.append(f"@section {stitle}\n\n")
                body = re.sub(r"\n{3,}", "\n\n", q["body"]).strip()
                if body:
                    o.append(body + "\n\n")
        o.append("@bye\n")
        with open(prefix + ".texi", "w", encoding="latin-1") as f:
            f.write("".join(o))


def _int_or(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def truthy(s):
    # Perl truthiness for the section number string: '' and '0' are false.
    return s not in ("", "0", 0, None)


def sanitise(s):
    out = ""
    i = 0
    for m in re.finditer(r'[<>&"]', s):
        out += s[i : m.start()] + "&" + SANI[m.group(0)] + ";"
        i = m.end()
    out += s[i:]
    return out


# ---------------------------------------------------------------------------
# driver (bfnnconv.pl)
# ---------------------------------------------------------------------------
def call(fnbase, *args):
    for be in outputs:
        if fnbase == "text" and be.cmds:
            be.args[-1] += " ".join(args)
        else:
            getattr(be, fnbase)(*args)


def recurse(be, s):
    global outputs, holdover
    save_o = outputs
    save_h = holdover
    outputs = [be]
    holdover = ""
    text(s)
    outputs = save_o
    holdover = save_h


# The BFNN inline-markup tokenizer (the Perl original's substitution loop): a
# flat sequence of prioritized regex rules over the running text, clearest read
# top-to-bottom as the ordered grammar it is, so its branch and statement
# counts are expected.
def text(s):  # pylint: disable=too-many-branches,too-many-statements
    global holdover
    inp = holdover + s
    holdover = ""
    while "\\" in inp:
        idx = inp.index("\\")
        before = inp[:idx]
        rhs = inp[idx + 1 :]
        call("text", before)
        cur = rhs
        m = re.match(r"^\w+ $", cur, A)
        if m:
            holdover = "\\" + m.group(0)
            inp = ""
            continue
        m = re.match(r"^fn\s+([^\s\\]*\w)", cur, A)
        if m:
            inp = cur[m.end() :]
            word = m.group(1)
            call("courier")
            call("text", word)
            call("endcourier")
            continue
        m = re.match(r"^tab\s+(\d+)\s+", cur, A)
        if m:
            inp = cur[m.end() :]
            call("tab", m.group(1))
            continue
        m = re.match(r"^nl\s+", cur, A)
        if m:
            inp = cur[m.end() :]
            call("newline")
            continue
        m = re.match(r"^qref\s+(\w+)", cur, A)
        if m:
            ref = m.group(1)
            refn = qrefn.get(ref, "")
            reft = qreft.get(ref, "")
            if not refn:
                sys.stderr.write(f"unknown question `{ref}'\n")
            post = cur[m.end() :]
            inp = f"\\pageref:{ref}:{refn}:{reft}\\endpageref." + post
            continue
        m = re.match(r"^pageref:(\w+):([^:\n]+):", cur, A)
        if m:
            inp = cur[m.end() :]
            call("pageref", m.group(1), m.group(2))
            continue
        m = re.match(r"^endpageref\.", cur, A)
        if m:
            inp = cur[m.end() :]
            call("endpageref")
            continue
        m = re.match(r"^(\w+)\{", cur, A)
        if m:
            inp = cur[m.end() :]
            fn = m.group(1)
            try:
                call(fn)
            except AttributeError as ex:
                sys.stderr.write(str(ex) + "\n")
                fn = "x"
            styles.append(fn)
            continue
        m = re.match(r"^\}", cur, A)
        if m:
            inp = cur[m.end() :]
            fn = styles.pop()
            if fn != "x":
                call("end" + fn)
            continue
        m = re.match(r"^\\", cur, A)
        if m:
            inp = cur[m.end() :]
            call("text", "\\")
            continue
        m = re.match(r"^(\w+)\s+([-A-Za-z0-9.@:/]*\w)", cur, A)
        if m:
            inp = cur[m.end() :]
            style = m.group(1)
            word = m.group(2)
            call(style)
            call("text", word)
            call("end" + style)
            continue
        sys.stderr.write(f"unknown control `\\{cur}'\n")
        inp = cur
    call("text", inp)


def process_copyto(fh_name, lines, idx):
    """Handle a \\copyto block: raw passthrough with backtick substitution."""
    be = backends.get(fh_name.lower())
    if be is not None and be not in outputs:
        be = None  # \copyto to an inactive backend (e.g. ASCII under -only texi)
    out = []
    while idx < len(lines):
        line = lines[idx]
        idx += 1
        if re.match(r"^\\endcopy$", line):
            break
        s = line + "\n"
        # process backtick command substitution
        while True:
            m = re.match(r"^([^`]*)`", s)
            if not m:
                break
            out.append(m.group(1))
            rest = s[m.end() :]
            m2 = re.search(r"([^\\])`", rest)
            if not m2:
                sys.stderr.write(f"`{rest}'\n")
                s = rest
                break
            cmd = rest[: m2.start()] + m2.group(1)
            s = rest[m2.end() :]
            pm = re.match(r"^%perl ", cmd)
            if pm:
                it = eval_perl_date(cmd[pm.end() :])
            else:
                it = os.popen(cmd).read()
                if it.endswith("\n"):
                    it = it[:-1]
            out.append(it)
        out.append(s)
    if be is not None:
        be.buf.append("".join(out))
    return idx


def eval_perl_date(expr):
    m = re.search(r'strftime\("([^"]*)"', expr)
    fmt = m.group(1) if m else "%Y-%m-%d"
    return strftime_utc(fmt)


def read_xrefdb(fname):
    global maxsection
    try:
        with open(fname, encoding="latin-1") as f:
            lines = f.readlines()
    except OSError as e:
        sys.stderr.write(f"no {fname} ({e})\n")
        return
    for thisxr in lines:
        s = thisxr.rstrip("\n")
        m = re.match(r"^Q (\w+) ((\d+)\.(\d+)) (.*)$", s, A)
        if m:
            qrefn[m.group(1)] = m.group(2)
            qreft[m.group(1)] = m.group(5)
            qn2ref[(m.group(3), m.group(4))] = m.group(1)
            maxsection = int(m.group(3))
            maxquestion[int(m.group(3))] = int(m.group(4))
            continue
        m = re.match(r"^S (\d+) ", s, A)
        if m:
            maxsection = int(m.group(1))
            sn2title[int(m.group(1))] = s[m.end() :]


def section_index(sec):
    """Emit the per-section question index that follows a \\section heading."""
    if not sec:
        return
    call("endpara")
    call("startindex")
    for entry in xrefdb_snapshot:
        mq = re.match(r"^Q (\w+) (\d+)\.(\d+) (.*)$", entry.rstrip("\n"), A)
        if not mq or int(mq.group(2)) != sec:
            continue
        call(
            "startindexitem",
            mq.group(1),
            f"Q{mq.group(2)}.{mq.group(3)}",
            f"Question {mq.group(2)}.{mq.group(3)}",
        )
        text(mq.group(4))
        call("endindexitem")
    call("endindex")


def full_index():
    """Emit the global cross-reference index (the \\index directive)."""
    call("startindex")
    for entry in xrefdb_snapshot:
        s = entry.rstrip("\n")
        mq = re.match(r"^Q (\w+) (\d+\.\d+) (.*)$", s, A)
        if mq:
            call(
                "startindexitem",
                mq.group(1),
                f"Q{mq.group(2)}",
                f"Question {mq.group(2)}",
            )
            text(mq.group(3))
            call("endindexitem")
            continue
        ms = re.match(r"^S (\d+) (.*)$", s, A)
        if ms:
            if not int(ms.group(1)):
                continue
            call(
                "startindexmainitem",
                f"s_{ms.group(1)}",
                f"Section {ms.group(1)}.",
                f"Section {ms.group(1)}",
            )
            text(ms.group(2))
            call("endindexitem")
            continue
        sys.stderr.write(s + "\n")
    call("endindex")


def question_links(sec, q):
    """Return the (next, previous) navigation labels for a question heading."""
    if q < maxquestion.get(sec, 0):
        nxt = f"Question {sec}.{q + 1}"
    elif sec < maxsection:
        nxt = f"Question {sec + 1}.1"
    else:
        nxt = ""
    if q > 1:
        prev = f"Question {sec}.{q - 1}"
    elif sec > 1:
        prev = f"Question {sec - 1}.{maxquestion.get(sec - 1, 0)}"
    else:
        prev = "Top"
    return nxt, prev


def read_verbatim(lines, i):
    """Emit a \\verbatim ... \\endverbatim block; return the new line index."""
    call("startverbatim")
    while i < len(lines):
        vline = lines[i]
        i += 1
        if re.match(r"^\\endverbatim$", vline):
            break
        call("verbatim", vline)
    call("endverbatim")
    return i


def handle_section(ln):
    """Open a new \\section: bump the counter, record it and emit its heading."""
    global section, question
    section += 1
    question = 0
    U.append(f"S {section} {ln}\n")
    call("endpara")
    call(
        "startmajorheading",
        f"{section}",
        f"Section {section}",
        f"Section {section + 1}" if section < maxsection else "",
        f"Section {section - 1}" if section > 1 else "Top",
    )
    text(ln)
    call("endmajorheading")
    section_index(section)


def handle_question(m, ln):
    """Open a new \\question: bump the counter and emit its heading."""
    global question
    question += 1
    qrefstring = m.group(1)
    if qrefstring.startswith(":"):
        qrefstring = qrefstring[1:]
    else:
        qrefstring = f"q_{section}_{question}"
    U.append(f"Q {qrefstring} {section}.{question} {ln}\n")
    call("endpara")
    nxt, prev = question_links(section, question)
    call(
        "startminorheading",
        qrefstring,
        f"Question {section}.{question}",
        nxt,
        prev,
        f"Section {section}",
    )
    text(f"Question {section}.{question}.  {ln}")
    call("endminorheading")


def process_line(line, lines, i):  # pylint: disable=too-many-return-statements
    """Dispatch one source line to its handler; return the next line index.

    ``i`` already points past ``line``; the \\copyto and \\verbatim handlers
    consume further lines and return the updated index.  This is a flat
    dispatch table -- one guard clause per BFNN line verb, matched in priority
    order -- so the return-per-verb count is expected.
    """
    if re.match(r"^\\comment\b", line):
        return i
    if not re.search(r"\S", line):
        call("endpara")
        return i
    m = re.match(r"^\\section +", line)
    if m:
        handle_section(line[m.end() :])
        return i
    m = re.match(r"^\\question \d{2}[a-z]{3}((:\w+)?) +", line, A)
    if m:
        handle_question(m, line[m.end() :])
        return i
    m = re.match(r"^\\only +", line)
    if m:
        handle_only(line[m.end() :])
        return i
    if re.match(r"^\\endonly$", line):
        restore_only()
        return i
    m = re.match(r"^\\copyto +", line)
    if m:
        return process_copyto(line[m.end() :], lines, i)
    if re.search(r"\\index$", line):
        full_index()
        return i
    m = re.match(r"^\\set +(\w+)\s*", line, A)
    if m:
        user[m.group(1)] = line[m.end() :]
        return i
    if re.match(r"^\\verbatim$", line):
        return read_verbatim(lines, i)
    # default: paragraph text
    line = re.sub(r"\.$", ". ", line)
    text(line + " ")
    return i


def main():
    global prefix, backends, outputs, section, question, holdover

    args = sys.argv[1:]
    only = None
    while args and args[0].startswith("-"):
        opt = args.pop(0)
        if opt.startswith("-only"):
            only = [args.pop(0)]
        else:
            sys.stderr.write(f"unknown option `{opt}' ignored\n")

    prefix = args[0] if args else "stdin"
    prefix = re.sub(r"\.bfnn$", "", prefix)

    read_xrefdb(prefix + ".xrefdb")

    backends = {"ascii": Ascii(), "info": Info(), "html": Html(), "texi": Texi()}
    order = only if only else ["ascii", "info", "html"]
    outputs = [backends[n] for n in order]

    with open(args[0], encoding="latin-1") as f:
        lines = f.read().split("\n")
    # Perl's <> drops the trailing newline per line; a final empty field from
    # split is spurious if the file ends in newline.
    if lines and lines[-1] == "":
        lines.pop()

    call("init")

    section = -1
    question = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        i = process_line(line, lines, i)

    call("finish")

    # write xrefdb-new and rename
    with open(prefix + ".xrefdb-new", "w", encoding="latin-1") as f:
        f.write("".join(U))
    os.replace(prefix + ".xrefdb-new", prefix + ".xrefdb")

    # write output files (only for the active backends)
    for be in outputs:
        be.write_files()


saveoutputs = []


def handle_only(rest):
    global outputs, saveoutputs
    saveoutputs = outputs
    names = [x for x in re.split(r"\s+", rest) if x]
    outputs = [be for be in saveoutputs if be.name in names]


def restore_only():
    global outputs
    outputs = saveoutputs


# xrefdb_snapshot: the Perl reads @xrefdb (the *previous* pass's file) for the
# \section and \index loops.  We captured it at read time.
xrefdb_snapshot = []


def _load_snapshot(fname):
    global xrefdb_snapshot
    try:
        with open(fname, encoding="latin-1") as f:
            xrefdb_snapshot = f.readlines()
    except OSError:
        xrefdb_snapshot = []


if __name__ == "__main__":
    # snapshot must reflect the previous pass, before we overwrite it
    pfx = sys.argv[-1]
    pfx = re.sub(r"\.bfnn$", "", pfx)
    _load_snapshot(pfx + ".xrefdb")
    main()
