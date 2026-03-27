PAPER = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex
	$(LATEX) $(PAPER)
	$(LATEX) $(PAPER)

clean:
	rm -f $(PAPER).pdf $(PAPER).aux $(PAPER).log $(PAPER).out \
	      $(PAPER).bbl $(PAPER).blg $(PAPER).toc $(PAPER).lof \
	      $(PAPER).lot $(PAPER).fls $(PAPER).fdb_latexmk

view: $(PAPER).pdf
	xdg-open $(PAPER).pdf 2>/dev/null || open $(PAPER).pdf 2>/dev/null
